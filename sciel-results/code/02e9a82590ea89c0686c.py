import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1, 16]
ALL_SETS = [frozenset(c) for c in combinations(range(1, 17), 4)]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_family = []
    best_size = 0
    
    # Use a deterministic seed for reproducibility of the "first" run, 
    # but we want to explore different orders.
    # The task evaluator might run this multiple times or just once.
    # Usually, these tasks want the best static output or a stochastic search.
    # "Maximize family cardinality" implies we should search.
    
    # Strategy: Multiple random greedy passes.
    # To ensure we find a good solution, we iterate until deadline.
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
            
        # Shuffle the candidate sets
        candidates = ALL_SETS[:]
        random.shuffle(candidates)
        
        family = []
        
        # Optimization: 
        # We check validity of S against current family.
        # We use the bucketing technique based on intersection with S.
        
        for S in candidates:
            # Check if S can be added
            # Bucket family members by their intersection with S
            buckets = {}
            for A in family:
                inter = A & S
                if inter not in buckets:
                    buckets[inter] = []
                buckets[inter].append(A)
            
            conflict = False
            # For each bucket, check if any pair A, B has A & B == inter
            for inter, members in buckets.items():
                m_len = len(members)
                if m_len < 2:
                    continue
                
                # Check pairs in this bucket
                # Note: We only need to check if A & B == inter
                for i in range(m_len):
                    A = members[i]
                    for j in range(i+1, m_len):
                        B = members[j]
                        if A & B == inter:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                family.append(S)
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = family[:]
                    
        # If we found a very large family, we might stop early?
        # No, keep searching for larger ones.
        
    return [sorted(list(s)) for s in best_family]
