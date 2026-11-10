import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Deterministic Construction: Modular Grid
    # We want a set S such that all sums a+b (a<=b) are distinct or appear at most twice.
    # A known construction for Sidon sets (max 2 reps) is based on finite fields or modular arithmetic.
    # For a set {i * N + j}, sums are i1N+j1 + i2N+j2 = (i1+i2)N + (j1+j2).
    # If we restrict i and j ranges such that j1+j2 < N, the representation is unique.
    # Let's try a "Bose-Chowla" style construction on Z_p.
    # Actually, a simpler deterministic base:
    # Let N = 32.
    # S = { i * 33 + j for i in 0..29, j in 1..32 }? 
    # Sum: (i1+i2)*33 + (j1+j2).
    # Max sum of j is 32+32=64 > 33. Collision possible if j1+j2 = j3+j4 + 33?
    # No, we need uniqueness of (i_sum, j_sum) modulo something?
    # If j_sum < 33, then i_sum is unique.
    # So we need j1+j2 < 33.
    # Max j is 32. If we pick j from 1..16, max sum is 32 < 33.
    # So S = { i * 33 + j for i in 0..k, j in 1..16 }.
    # Size = (k+1) * 16.
    # Max element: k*33 + 16 <= 1000.
    # k*33 <= 984 => k <= 29.8. So k=29.
    # Size = 30 * 16 = 480.
    # Let's verify this is a Sidon set (distinct sums).
    # If a1+a2 = a3+a4, then (i1+i2)33 + (j1+j2) = (i3+i4)33 + (j3+j4).
    # Since 0 < j1,j2,j3,j4 <= 16, we have 2 <= j1+j2 <= 32 and 2 <= j3+j4 <= 32.
    # The difference (j1+j2) - (j3+j4) is in [-30, 30].
    # For the equality to hold, (i1+i2) - (i3+i4) must be 0, because 33 > 30.
    # So i1+i2 = i3+i4 and j1+j2 = j3+j4.
    # This allows multiple representations! e.g., {1, 2} and {1.5, 1.5} no, integers.
    # {1, 3} sum 4, {2, 2} sum 4.
    # So this is NOT a Sidon set. It's a set where sums are determined by sums of components.
    # The number of representations of a sum s is the number of ways to write s as i_sum*33 + j_sum.
    # For a fixed i_sum, j_sum is fixed.
    # The number of pairs (i1, i2) with i1+i2 = I is at most 15 (for I=16, pairs 0+16..16+0, but i in 0..29).
    # Wait, i ranges 0..29.
    # Max i_sum = 58.
    # Number of solutions to i1+i2=I with 0<=i1,i2<=29.
    # For I=16, solutions: 0+16, 1+15, ..., 16+0 (17 solutions).
    # Plus j1+j2=J.
    # If J=4, solutions for j (1..16): 1+3, 2+2, 3+1 (3 solutions).
    # Total reps for sum = 17 * 3 = 51.
    # This violates B2[2] (max 2 reps).
    
    # We need a construction with at most 2 representations.
    # Sidon sets (B2[1]) have 1 rep. B2[2] allows 2.
    # Random sets of size ~ N/sqrt(N) = sqrt(N) are likely Sidon.
    # For N=1000, sqrt(1000) ~ 31.
    # We can get much larger for B2[2].
    # The density of Sidon sets is ~ N^{1/2}.
    # The density of B2[2] sets is ~ N^{1/2} * constant.
    # Actually, for B2[k], the max size is ~ N^{1/2} * k^{1/2}? No.
    # Let's just use a greedy randomized approach but with a faster check.
    # The previous program was slow because of Python loops.
    # We can use NumPy to check conflicts in bulk?
    # Or we can use a smarter greedy:
    # Pick numbers in increasing order? No, random is better for spread.
    # Let's try to construct a set of size ~ 200-300.
    
    # Strategy: 
    # 1. Start with empty set.
    # 2. Repeatedly pick a random candidate x not in S.
    # 3. Check if adding x creates a sum with > 2 reps.
    #    A sum s = x + y. 
    #    Existing reps of s in S: count_s.
    #    New reps involving x: 
    #      x + x = 2x.
    #      x + y for y in S.
    #    We need count_s + new_reps_s <= 2 for all s.
    #    This is hard to check in bulk easily without matrix ops.
    #    
    # Alternative: Use the property that if S is Sidon, it's B2[1].
    # We can try to build a Sidon set first (size ~ 30-40), then try to add elements that "share" sums.
    # 
    # Let's try a different angle: 
    # Use the construction S = { a * 100 + b } where a, b are small.
    # If we restrict a to {0, 1} and b to {1..50}, size 100.
    # Sums: (a1+a2)*100 + (b1+b2).
    # b1+b2 max 100. Collision if b1+b2 = b3+b4 + 100? No, max 100.
    # So b_sum determines b1+b2 uniquely? No, 1+50=51, 26+25=51.
    # Reps of b_sum: 
    # If b_sum < 51, reps = b_sum - 1.
    # If b_sum >= 51, reps = 101 - b_sum.
    # Max reps for b_sum is 50 (at 51).
    # a_sum determines a1+a2.
    # If a in {0,1}, a_sum is 0,1,2.
    # Reps of a_sum: 
    # 0: (0,0) - 1 rep.
    # 1: (0,1), (1,0) - 2 reps.
    # 2: (1,1) - 1 rep.
    # Total reps for total sum = reps(a_sum) * reps(b_sum).
    # We need total <= 2.
    # So we need reps(a_sum) * reps(b_sum) <= 2.
    # This restricts b_sum to have <= 1 rep if a_sum has 2 reps, or <= 2 reps if a_sum has 1 rep.
    # This is very restrictive.
    
    # Let's go back to randomized greedy but optimize the check.
    # The check is: for all y in S, count[x+y] < 2 AND count[2x] < 2.
    # We can maintain an array `counts` of size 2001.
    # When adding x, we check all y in S.
    # If S is size M, this is O(M).
    # If we try 1000 candidates, O(1000*M).
    # If M=200, O(200,000) ops per restart.
    # We can do many restarts in 120s.
    
    best_set = []
    best_size = 0
    
    def try_greedy(order):
        S = set()
        counts = np.zeros(2001, dtype=np.int8)
        
        for x in order:
            # Check if x can be added
            # 1. Check 2x
            if counts[2*x] >= 2:
                continue
            # 2. Check x+y for y in S
            ok = True
            # Optimization: if S is large, this is slow.
            # But we can break early.
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    ok = False
                    break
            if ok:
                S.add(x)
                counts[2*x] += 1
                for y in S:
                    if y != x:
                        counts[x+y] += 1
        return S
    
    # Run multiple random restarts
    for i in range(50):
        if time.time() > deadline - 5:
            break
        random.seed(i)
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        S = try_greedy(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # Print progress? No, just keep it.
    
    # Local search on the best set found
    if best_size > 0:
        S = set(best_set)
        counts = np.zeros(2001, dtype=np.int8)
        for x in S:
            counts[2*x] += 1
            for y in S:
                if y != x:
                    counts[x+y] += 1
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try to add a random element
            for _ in range(200):
                x = random.randint(1, 1000)
                if x in S:
                    continue
                if counts[2*x] >= 2:
                    continue
                ok = True
                for y in S:
                    if counts[x+y] >= 2:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    counts[2*x] += 1
                    for y in S:
                        if y != x:
                            counts[x+y] += 1
                    improved = True
                    break
            
            if not improved:
                # Try to swap
                # Remove a random x, try to add a different y
                # This is complex to manage counts efficiently in Python loop.
                # Let's just do a few random restarts if we have time.
                pass
                
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
