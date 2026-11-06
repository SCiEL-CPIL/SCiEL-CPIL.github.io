import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    def is_3ap_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b: continue
                # Check if 2*a = b + c for some c in S
                # c = 2*a - b
                if 2*a - b in S_set and 2*a - b != b:
                    return False
        return True

    def has_4_sidon_subset(A):
        # A is a list of 12 ints.
        # Check if any 4-element subset is a Sidon set.
        # Since A is assumed 3-AP-free, we only need to check for a+b = c+d with {a,b} != {c,d}.
        # If such a collision exists, the subset is NOT Sidon.
        # If no such collision exists for a specific 4-subset, it IS Sidon.
        # We want to find if ANY 4-subset is Sidon.
        # If we find one, return True.
        # If we check all and find none, return False.
        
        for subset in combinations(A, 4):
            sums = set()
            is_sidon = True
            # Check distinct pairs
            # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            s01 = subset[0] + subset[1]
            s02 = subset[0] + subset[2]
            s03 = subset[0] + subset[3]
            s12 = subset[1] + subset[2]
            s13 = subset[1] + subset[3]
            s23 = subset[2] + subset[3]
            
            if len({s01, s02, s03, s12, s13, s23}) < 6:
                # Collision exists, so NOT Sidon
                continue
            # If all 6 sums are distinct, and no APs (guaranteed by 3-AP-free ambient), 
            # then it is a Sidon set.
            return True
        
        return False

    def gen_random_3ap_free(k, limit, rng):
        # Greedy with randomization
        S = []
        # Start with 0 and 1 to ensure distinctness and small values
        # Actually, just pick random valid elements
        # To make it faster, we can use a greedy approach with random ordering of candidates
        
        # Let's try a simple random greedy
        # We want to pick k elements from [0, limit]
        
        # A better way:
        # Start with empty set.
        # Repeatedly pick a random number from [0, limit] that doesn't form an AP with existing pairs.
        # If stuck, restart.
        
        for _ in range(100): # Max attempts
            S = []
            # Pick first two randomly? Or 0,1?
            # 0,1 is safe.
            S.append(0)
            if k > 1: S.append(1)
            
            # Pool of candidates
            # To ensure we can find k elements, we should pick from a wide range.
            # [0, limit]
            
            while len(S) < k:
                # Try to add a new element
                found = False
                # Shuffle candidates to add randomness
                # Candidates are all x in [0, limit] not in S
                # This is O(limit) per step. Limit=128, k=12. 12*128 is small.
                candidates = [x for x in range(limit + 1) if x not in S]
                rng.shuffle(candidates)
                
                for x in candidates:
                    # Check if adding x creates an AP
                    ok = True
                    for a in S:
                        for b in S:
                            if a == b: continue
                            # APs involving x:
                            # 1. x is middle: 2*x = a + b
                            # 2. x is end: 2*a = x + b  => x = 2*a - b
                            # 3. x is end: 2*b = x + a  => x = 2*b - a
                            
                            if 2*x == a + b:
                                ok = False
                                break
                            if x == 2*a - b and x != b: # x!=b is redundant if x not in S, but 2a-b could be b if a=b
                                ok = False
                                break
                            if x == 2*b - a and x != a:
                                ok = False
                                break
                        if not ok: break
                    
                    if ok:
                        S.append(x)
                        found = True
                        break
                
                if not found:
                    break
            
            if len(S) == k:
                return S
        
        return None

    best_set = None
    best_score = 999
    
    rng = random.Random(42)
    
    # We want score 3.
    # Try to find a 3-AP-free set with no 4-Sidon subsets.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        S = gen_random_3ap_free(12, 128, rng)
        if S is None:
            continue
        
        # Verify it is actually 3-AP free (safety)
        if not is_3ap_free(S):
            continue
        
        # Check for 4-Sidon subsets
        if not has_4_sidon_subset(S):
            # Score is 3
            best_set = S
            break
            
        # If we didn't find one, we might need to search harder.
        # But random 3-AP-free sets might often have 4-Sidon subsets.
        # Let's see if we can do local search to remove 4-Sidon subsets.
        
    if best_set is None:
        # Fallback: return any 3-AP-free set (score >= 3, likely 4)
        # Or just a dense set (score ~ 4-5)
        # But we should try to return the best we found.
        # If we found nothing with score 3, we return a 3-AP-free set (score 3 or 4).
        # Actually, the loop above doesn't keep the best 3-AP-free set if it fails the 4-Sidon check.
        # Let's modify to keep the best candidate.
        
        # Re-run with tracking
        best_candidate = None
        best_candidate_score = 999
        
        for _ in range(1000): # Fixed number of attempts if time is tight, or loop again
            if time.time() > deadline: break
            S = gen_random_3ap_free(12, 128, rng)
            if S is None: continue
            if not is_3ap_free(S): continue
            
            # Calculate exact score
            # We know it's at least 3.
            # Check if it has 4-Sidon.
            if not has_4_sidon_subset(S):
                best_candidate = S
                best_candidate_score = 3
                break
            else:
                if best_candidate is None:
                    best_candidate = S
                    best_candidate_score = 4 # It has a 4-Sidon subset. Might have 5?
                    # We should check for 5-Sidon too to be sure of the score, 
                    # but for the purpose of "minimizing", finding a 3 is best.
                    # If we can't find 3, a 4 is better than a 5.
                    # So we should estimate the score.
                    # has_4_sidon_subset returns True if ANY 4-subset is Sidon.
                    # If so, score >= 4.
                    # To know if score is 4 or >4, we'd need to check 5-subsets.
                    # C(12,5) = 792. Checking each is more expensive.
                    # Let's just assume if it has a 4-Sidon, it's "worse" than one that doesn't.
        
        if best_candidate:
            best_set = best_candidate
        else:
            # Absolute fallback
            best_set = list(range(12))

    return best_set
