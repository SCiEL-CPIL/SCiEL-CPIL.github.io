import numpy as np
import time
import random

# Precompute 4-term arithmetic progressions
# A 4-AP is defined by (a, d) such that a, a+d, a+2d, a+3d are in [0, 127]
# Indices are 0-based.
def get_aps():
    aps = []
    n = 128
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return np.array(aps)

APs = get_aps()

def count_bad(colors):
    """
    Colors is a numpy array of shape (128,) with integer values.
    Returns the number of 4-APs that use fewer than 3 distinct colors.
    """
    c = colors[APs]  # Shape (M, 4)
    
    # A 4-AP is bad if it has 1 or 2 distinct colors.
    # 1 distinct: all 4 are equal.
    bad_1 = np.all(c == c[:, 0:1], axis=1)
    
    # 2 distinct:
    # Patterns:
    # A A A B
    # A A B B
    # A B A B
    # A B B A
    # A B B B
    # B A A A
    # B A A B
    # B A B A
    # B B A A
    # B B A B
    # B B B A
    # Basically, if it's not 1 distinct, it's 2 distinct if it doesn't have 3 or 4 distinct.
    # Easier: Check if the number of unique values is 2.
    # We can check specific equality patterns that imply <= 2 unique values.
    # If any of these patterns hold, and it's not 1 distinct (which is a subset of some), it's bad.
    
    # Let's define "bad" as:
    # (c0==c1 & c1==c2 & c2==c3)  # 1 color
    # OR (c0==c1 & c1==c2 & c2!=c3) # AAA B
    # OR (c0==c1 & c1!=c2 & c2==c3) # AA BB
    # OR (c0==c2 & c1==c3)          # AB AB
    # OR (c0==c3 & c1==c2)          # AB BA
    # OR (c0!=c1 & c1==c2 & c2==c3) # A BBB
    # OR (c0==c1 & c1!=c2 & c2!=c3 & c0==c3) # A B A B? No, that's ABAB.
    # Wait, the above patterns cover all cases with <= 2 distinct values.
    # Let's verify.
    # If 2 distinct values, say A and B.
    # The sequence is a string of A and B of length 4.
    # If it has 3 A's and 1 B: AAA B, A A B A, A B A A, B A A A.
    #   AAA B: c0=c1=c2, c2!=c3.
    #   A A B A: c0=c1, c1!=c2, c2!=c3, c0=c3. (Pattern: c0=c1 & c0=c3)
    #   A B A A: c0=c2, c2=c3, c1!=c0. (Pattern: c0=c2 & c2=c3)
    #   B A A A: c1=c2, c2=c3, c0!=c1. (Pattern: c1=c2 & c2=c3)
    # If it has 2 A's and 2 B's:
    #   A A B B: c0=c1, c2=c3.
    #   A B A B: c0=c2, c1=c3.
    #   A B B A: c0=c3, c1=c2.
    #   B A A B: c0=c3, c1=c2 (Same as ABBA with A/B swapped? No. B A A B: c0=B, c1=A, c2=A, c3=B. c0=c3, c1=c2).
    #   B A B A: c0=c2, c1=c3.
    #   B B A A: c0=c1, c2=c3.
    # If it has 1 A and 3 B's: Symmetric to 3 A's.
    
    # So the conditions for 2 distinct (and not 1) are:
    # 1. c0==c1 and c1==c2 and c2!=c3
    # 2. c0==c1 and c1!=c2 and c2==c3
    # 3. c0==c2 and c1==c3
    # 4. c0==c3 and c1==c2
    # 5. c0!=c1 and c1==c2 and c2==c3
    # 6. c0==c1 and c1!=c2 and c2!=c3 and c0==c3  (A B A B is covered by 3? A B A B: c0=A, c2=A. c1=B, c3=B. Yes, c0=c2 and c1=c3. So ABAB is covered by 3.)
    #    Wait, A B A B: c0=c2, c1=c3. Yes.
    #    What about A A B A? c0=A, c1=A, c2=B, c3=A.
    #    c0=c1 (True), c1!=c2 (True), c2!=c3 (True), c0=c3 (True).
    #    Is it covered?
    #    Pattern 2: c0=c1 and c1!=c2 and c2=c3? No, c2=B, c3=A.
    #    Pattern 3: c0=c2? No.
    #    Pattern 4: c0=c3? Yes. c1=c2? No.
    #    So AABA is NOT covered by the simple patterns above?
    #    Let's check AABA distinct count: {A, B}. 2 distinct. Bad.
    #    My patterns missed AABA, ABAA, BAAB, BABA?
    #    Let's list all 2-color strings of length 4 that are not constant:
    #    AABA, ABAA, ABBA, ABAB, BAAB, BABA, BABB, BBAA, BBAB, BBBA? No, 3 B's.
    #    Let's just use a robust check.
    #    Since M is small, we can just check if the number of unique elements is < 3.
    #    `np.unique` is slow.
    #    But we can check:
    #    bad = (c[:,0]==c[:,1]) & (c[:,1]==c[:,2]) & (c[:,2]==c[:,3]) # 1
    #    | (c[:,0]==c[:,1]) & (c[:,1]==c[:,2]) & (c[:,2]!=c[:,3]) # AAA B
    #    | (c[:,0]==c[:,1]) & (c[:,1]!=c[:,2]) & (c[:,2]==c[:,3]) # AA BB
    #    | (c[:,0]==c[:,2]) & (c[:,1]==c[:,3]) # AB AB
    #    | (c[:,0]==c[:,3]) & (c[:,1]==c[:,2]) # AB BA
    #    | (c[:,0]!=c[:,1]) & (c[:,1]==c[:,2]) & (c[:,2]==c[:,3]) # A BBB
    #    | (c[:,0]==c[:,1]) & (c[:,1]!=c[:,2]) & (c[:,2]!=c[:,3]) & (c[:,0]==c[:,3]) # A B A B? No, AABA.
    #       AABA: c0=A, c1=A, c2=B, c3=A.
    #       c0=c1 (T), c1!=c2 (T), c2!=c3 (T), c0=c3 (T).
    #       This pattern: c0=c1 & c0=c3 & c1!=c2.
    #    | (c[:,0]!=c[:,1]) & (c[:,1]==c[:,2]) & (c[:,2]!=c[:,3]) & (c[:,1]==c[:,3]) # B A A B?
    #       B A A B: c0=B, c1=A, c2=A, c3=B.
    #       c0!=c1 (T), c1=c2 (T), c2!=c3 (T), c1!=c3 (F).
    #       Pattern: c1=c2 & c0=c3. (Covered by AB BA pattern? AB BA is A B B A. c0=A, c1=B, c2=B, c3=A. c0=c3, c1=c2. Yes. B A A B is c0=B, c1=A, c2=A, c3=B. c0=c3, c1=c2. Yes, covered by Pattern 4).
    #       
    #    Let's check ABAA: A B A A.
    #    c0=A, c1=B, c2=A, c3=A.
    #    c0=c2 (T), c2=c3 (T), c1!=c0 (T).
    #    Pattern: c0=c2 & c2=c3.
    #    
    #    So the missing patterns are:
    #    AABA: c0=c1 & c0=c3 & c1!=c2
    #    ABAA: c0=c2 & c2=c3 & c1!=c0
    #    BAAB: c1=c2 & c0=c3 & c0!=c1 (Covered by c0=c3 & c1=c2? No. BAAB: c0=B, c1=A, c2=A, c3=B. c0=c3, c1=c2. Yes, covered by Pattern 4).
    #    BABA: c0=c2 & c1=c3. Covered by Pattern 3.
    
    # So we need to add:
    # 7. (c[:,0]==c[:,1]) & (c[:,0]==c[:,3]) & (c[:,1]!=c[:,2]) # AABA
    # 8. (c[:,0]==c[:,2]) & (c[:,2]==c[:,3]) & (c[:,1]!=c[:,0]) # ABAA
    #    Note: AABA and ABAA are symmetric.
    #    Are there others?
    #    B A A B is covered.
    #    B B A A is covered (AA BB pattern with swap).
    #    B A B A is covered (AB AB pattern with swap).
    #    B A A B? Covered.
    #    What about B B A B? B B A B: c0=B, c1=B, c2=A, c3=B.
    #    c0=c1, c1!=c2, c2!=c3, c0=c3.
    #    This is the same pattern as AABA (just swap A and B).
    #    So Pattern 7 covers AABA and BBAB.
    #    Pattern 8 covers ABAA and BBAB? No, ABAA is A B A A. BBAB is B B A B.
    #    ABAA: c0=A, c1=B, c2=A, c3=A.
    #    BBAB: c0=B, c1=B, c2=A, c3=B.
    #    They are different.
    #    ABAA: c0=c2=c3.
    #    BBAB: c0=c1=c3.
    #    So we need:
    #    9. (c[:,0]==c[:,1]) & (c[:,1]==c[:,3]) & (c[:,0]!=c[:,2]) # BBAB (and AABA? No, AABA is c0=c1=c3? A A B A. c0=A, c1=A, c3=A. Yes! AABA is c0=c1=c3. Wait.
    #       AABA: A A B A. c0=A, c1=A, c2=B, c3=A.
    #       c0=c1 (T), c1=c3 (T), c0!=c2 (T).
    #       So AABA is covered by Pattern 9.
    #       What about ABAA? A B A A.
    #       c0=A, c1=B, c2=A, c3=A.
    #       c0=c2 (T), c2=c3 (T), c0!=c1 (T).
    #       This is Pattern 8.
    
    # So the complete set of bad patterns (2 distinct) is:
    # P1: c0=c1=c2, c2!=c3
    # P2: c0=c1, c1!=c2, c2=c3
    # P3: c0=c2, c1=c3
    # P4: c0=c3, c1=c2
    # P5: c0!=c1, c1=c2, c2=c3
    # P6: c0=c1, c1=c3, c0!=c2  (AABA, BBAB)
    # P7: c0=c2, c2=c3, c0!=c1  (ABAA, BBAB? No. ABAA is A B A A. c0=A, c2=A, c3=A. c0=c2=c3. c1=B. Yes.)
    
    # Let's combine them.
    bad_2 = (
        ((c[:,0]==c[:,1]) & (c[:,1]==c[:,2]) & (c[:,2]!=c[:,3])) |
        ((c[:,0]==c[:,1]) & (c[:,1]!=c[:,2]) & (c[:,2]==c[:,3])) |
        ((c[:,0]==c[:,2]) & (c[:,1]==c[:,3])) |
        ((c[:,0]==c[:,3]) & (c[:,1]==c[:,2])) |
        ((c[:,0]!=c[:,1]) & (c[:,1]==c[:,2]) & (c[:,2]==c[:,3])) |
        ((c[:,0]==c[:,1]) & (c[:,1]==c[:,3]) & (c[:,0]!=c[:,2])) |
        ((c[:,0]==c[:,2]) & (c[:,2]==c[:,3]) & (c[:,0]!=c[:,1]))
    )
    
    bad = bad_1 | bad_2
    return np.sum(bad)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit
    
    # Try to find a 3-coloring
    best_cost = float('inf')
    best_colors = None
    
    # Initial guess: Greedy
    greedy_colors = []
    for x in range(1, 129):
        for color in range(3): # Try 0, 1, 2
            if all(len({color} | {greedy_colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1)):
                greedy_colors.append(color)
                break
        else:
            # If no color works, force one (invalid intermediate)
            greedy_colors.append(0)
            
    greedy_colors = np.array(greedy_colors)
    cost_g = count_bad(greedy_colors)
    if cost_g == 0:
        return greedy_colors.tolist()
        
    best_colors = greedy_colors
    best_cost = cost_g
    
    # Simulated Annealing
    current_colors = greedy_colors.copy()
    current_cost = cost_g
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Random move
        i = random.randint(0, 127)
        old_c = current_colors[i]
        new_c = random.choice([c for c in range(3) if c != old_c])
        
        current_colors[i] = new_c
        new_cost = count_bad(current_colors)
        
        if new_cost < current_cost or (random.random() < np.exp(-(new_cost - current_cost) / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = current_colors.copy()
                if best_cost == 0:
                    break
        else:
            current_colors[i] = old_c
            
        T *= cooling
        if T < 0.01:
            # Restart
            T = 1.0
            current_colors = np.random.randint(0, 3, 128)
            current_cost = count_bad(current_colors)
            
    return best_colors.tolist()
