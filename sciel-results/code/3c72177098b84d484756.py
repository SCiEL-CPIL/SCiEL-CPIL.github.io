import random
import time
import math

def _calculate_score(S):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in S.
    S is a set of integers.
    """
    covered_d = set()
    # Iterate over all pairs (a, b) in S with a < b
    # Let d = b - a
    # Check if a + 2d is in S
    elements = list(S)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in S:
                covered_d.add(d)
    return len(covered_d)

def _greedy_construct():
    """
    Construct a set of 32 integers using a greedy heuristic.
    """
    S = set()
    # Start with a few random elements to break symmetry? 
    # Or just start empty and pick the best?
    # If S is empty, gain is 0. So we need to handle the first few picks specially or just pick randoms.
    # Let's pick 3 randoms to start, then greedily add.
    # Actually, any 3 distinct numbers form 0 3-APs with distinct d unless they are in AP.
    # Let's just start with 1, 2, 4 (powers of 2 tend to be good for distinct differences? No, that's for Sidon).
    # For many 3-APs, we want structure.
    # Let's just start with a random set of 5 and then greedily add.
    
    # Better: Start with 1. 
    # Actually, the "gain" of adding x to S is the number of new d's.
    # If S has < 2 elements, gain is always 0.
    # So we initialize with a small random set.
    
    S = set(random.sample(range(1, 513), 5))
    
    for _ in range(27):
        best_x = None
        best_gain = -1
        current_score = _calculate_score(S)
        
        # To speed up, we can pre-compute which d's are already covered?
        # But _calculate_score is fast.
        # Let's just evaluate each candidate.
        
        # Optimization: 
        # New d's come from:
        # 1. x, a, b where x = a - d, b = a + d (x is middle) -> d = b - a
        # 2. a, x, b where a = x - d, b = x + d (x is end) -> d = x - a
        # 3. a, b, x where a = x - 2d, b = x - d (x is end) -> d = b - a
        
        # Actually, simpler:
        # For each candidate x, new_d = set()
        # For each a in S:
        #   d1 = x - a
        #   if d1 > 0:
        #      if a + 2*d1 in S: new_d.add(d1) # x is a+2d
        #      if a - d1 in S: new_d.add(d1)   # x is a-d? No, a, a+d, a+2d. 
        #         If x = a - d, then a = x + d. Sequence: x, x+d, x+2d.
        #         So if x in S, and x+d in S, and x+2d in S.
        #         Here x is the new one.
        #         Case 1: x is the first term. x, x+d, x+2d. Need x+d, x+2d in S.
        #         Case 2: x is the second term. x-d, x, x+d. Need x-d, x+d in S.
        #         Case 3: x is the third term. x-2d, x-d, x. Need x-2d, x-d in S.
        
        # Let's just use the pair iteration method for score, it's robust.
        
        for x in range(1, 513):
            if x in S:
                continue
            S.add(x)
            new_score = _calculate_score(S)
            gain = new_score - current_score
            S.remove(x)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is not None:
            S.add(best_x)
        else:
            # Fallback: add random
            S.add(random.choice([x for x in range(1, 513) if x not in S]))
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    best_set = None
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.seed(num_restarts * 12345 + 678)
        
        # 1. Greedy Initialization
        S = _greedy_construct()
        current_score = _calculate_score(S)
        
        if current_score > best_score:
            best_score = current_score
            best_set = S.copy()
        
        # 2. Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        max_steps = 20000  # Adjust based on time
        
        S_list = list(S)
        S_set = S
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random element to remove
            i = random.randint(0, 31)
            x_out = S_list[i]
            
            # Pick a random element to add
            # To avoid infinite loops, pick from a pool or just random 1..512
            x_in = random.randint(1, 512)
            
            if x_in == x_out:
                continue
            
            if x_in in S_set:
                continue
            
            # Perform swap
            S_list[i] = x_in
            S_set.remove(x_out)
            S_set.add(x_in)
            
            new_score = _calculate_score(S_set)
            delta = new_score - current_score
            
            if delta >= 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = S_set.copy()
            else:
                # Accept with probability exp(-delta / T)
                # delta is negative, so -delta is positive
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    # Reject: revert
                    S_list[i] = x_out
                    S_set.remove(x_in)
                    S_set.add(x_out)
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
    return sorted(list(best_set))

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(len(s))
    print(s)
    print(_calculate_score(set(s)))
