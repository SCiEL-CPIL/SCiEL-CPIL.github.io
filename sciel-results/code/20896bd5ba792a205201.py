import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    def rebuild_diffs_sums(S):
        diffs = set()
        sums = set()
        for i, a in enumerate(S):
            sums.add(2 * a)
            for j in range(i + 1, len(S)):
                b = S[j]
                d = abs(a - b)
                diffs.add(d)
                sums.add(a + b)
        return diffs, sums

    def can_add(x, S, diffs_S, sums_S, diffs_other):
        # Check Sidon condition for S
        # 1. Check if 2x collides with existing sums
        if 2 * x in sums_S:
            return False
        
        # 2. Check if any new difference |x-a| collides with existing diffs in S
        #    or with diffs in the other set.
        for a in S:
            d = abs(x - a)
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
        
        return True

    def add_to_set(x, S, diffs_S, sums_S):
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)

    def do_local_search(A, B, max_iterations=2000):
        """
        Local search with Add, Move, and Kick.
        """
        diffs_A, sums_A = rebuild_diffs_sums(A)
        diffs_B, sums_B = rebuild_diffs_sums(B)
        A_set = set(A)
        B_set = set(B)
        
        best_local_score = len(A) + len(B)
        best_local_A = A.copy()
        best_local_B = B.copy()
        
        iterations = 0
        stall_count = 0
        kick_cooldown = 0
        
        while iterations < max_iterations:
            iterations += 1
            if time.time() > deadline:
                break
            
            made_improvement = False
            
            # 1. Try to Add
            # We try to add to the set that allows it. 
            # To be fast, we don't shuffle the entire candidate list every time.
            # We keep a list of candidates not in A or B.
            # But maintaining that list is overhead. 
            # Given N=256, iterating 1..256 is fast.
            
            # Shuffle a subset of candidates to introduce randomness
            # But for speed, we just iterate 1..256 in a random order generated once per 'stall' or just fixed?
            # Better: Keep a global random order for candidates, but try adding in that order.
            # If we get stuck, we reshuffle.
            
            # For simplicity and speed, let's just try to add any valid x from 1..256.
            # To avoid always picking the smallest, we can randomize the start point or use a random permutation.
            # Let's use a random permutation of 1..256 that is regenerated if we stall.
            
            # Actually, the current code shuffles candidates every iteration which is O(N log N) = 256*8 ~ 2000 ops.
            # 2000 ops * 2000 iters = 4M ops. Fine.
            
            candidates = [x for x in range(1, 257) if x not in A_set and x not in B_set]
            if not candidates:
                break # No more candidates to add
            
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if can_add(x, A, diffs_A, sums_A, diffs_B):
                    add_to_set(x, A, diffs_A, sums_A)
                    A_set.add(x)
                    made_improvement = True
                    added = True
                    break
                if can_add(x, B, diffs_B, sums_B, diffs_A):
                    add_to_set(x, B, diffs_B, sums_B)
                    B_set.add(x)
                    made_improvement = True
                    added = True
                    break
            
            if made_improvement:
                score = len(A) + len(B)
                if score > best_local_score:
                    best_local_score = score
                    best_local_A = A.copy()
                    best_local_B = B.copy()
                stall_count = 0
                continue
            
            # 2. If stuck, try a Move (A->B or B->A)
            moved = False
            # Try moving a random element from A to B
            if A:
                # Pick a random element from A to try moving
                # To be faster, don't try all, just a few random ones
                for _ in range(min(5, len(A))):
                    a = random.choice(A)
                    A_temp = [x for x in A if x != a]
                    diffs_A_temp, sums_A_temp = rebuild_diffs_sums(A_temp)
                    
                    if can_add(a, B, diffs_B, sums_B, diffs_A_temp):
                        # Valid move
                        A.remove(a)
                        A_set.discard(a)
                        diffs_A, sums_A = diffs_A_temp, sums_A_temp
                        
                        add_to_set(a, B, diffs_B, sums_B)
                        B_set.add(a)
                        
                        moved = True
                        break
                        # Don't break outer loop immediately, let it try to add next iter?
                        # Yes, continue to top of loop to try adding.
            
            if not moved and B:
                for _ in range(min(5, len(B))):
                    b = random.choice(B)
                    B_temp = [x for x in B if x != b]
                    diffs_B_temp, sums_B_temp = rebuild_diffs_sums(B_temp)
                    
                    if can_add(b, A, diffs_A, sums_A, diffs_B_temp):
                        B.remove(b)
                        B_set.discard(b)
                        diffs_B, sums_B = diffs_B_temp, sums_B_temp
                        
                        add_to_set(b, A, diffs_A, sums_A)
                        A_set.add(b)
                        
                        moved = True
                        break
            
            if moved:
                stall_count += 1
                continue
            
            # 3. If stuck and no moves, try a Kick
            # Remove 1-2 random elements and see if we can add more.
            if stall_count > 10 and kick_cooldown <= 0:
                # Kick: Remove 1 element from larger set
                if len(A) >= len(B) and A:
                    a = random.choice(A)
                    A.remove(a)
                    A_set.discard(a)
                    diffs_A, sums_A = rebuild_diffs_sums(A)
                elif B:
                    b = random.choice(B)
                    B.remove(b)
                    B_set.discard(b)
                    diffs_B, sums_B = rebuild_diffs_sums(B)
                
                kick_cooldown = 5
                stall_count = 0
                # After kick, we might be able to add something new
                continue
            
            kick_cooldown -= 1
            
            # 4. If nothing worked, break local search
            break
                
        return best_local_A, best_local_B, best_local_score

    def polish(A, B):
        """Final greedy add."""
        diffs_A, sums_A = rebuild_diffs_sums(A)
        diffs_B, sums_B = rebuild_diffs_sums(B)
        A_set = set(A)
        B_set = set(B)
        
        while True:
            if time.time() > deadline:
                break
            any_added = False
            # Try adding in random order
            candidates = [x for x in range(1, 257) if x not in A_set and x not in B_set]
            if not candidates:
                break
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, A, diffs_A, sums_A, diffs_B):
                    add_to_set(x, A, diffs_A, sums_A)
                    A_set.add(x)
                    any_added = True
                    break
                if can_add(x, B, diffs_B, sums_B, diffs_A):
                    add_to_set(x, B, diffs_B, sums_B)
                    B_set.add(x)
                    any_added = True
                    break
            if not any_added:
                break
        
        return A, B

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts * 98765 + 111)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Greedy Initialization
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                # Balance heuristic
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        # Local Search
        A, B, local_score = do_local_search(A, B)
        
        # Polish
        A, B = polish(A, B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()
            
        # Early stop if we hit a very high score?
        # Max Sidon size in [1, 256] is approx sqrt(256)=16.
        # So max total 32.
        if best_score >= 32:
            break

    return [best_A, best_B]
