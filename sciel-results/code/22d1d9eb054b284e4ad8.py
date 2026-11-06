import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1..16]
N = 16
K = 4
ALL_SUBSETS = list(combinations(range(1, N + 1), K))
NUM_SUBSETS = len(ALL_SUBSETS)  # C(16,4) = 1820

# Convert to bitmasks for fast intersection
def subset_to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

SUBSET_MASKS = [subset_to_mask(s) for s in ALL_SUBSETS]

# Precompute intersection sizes
# INTER[i][j] = |subset_i ∩ subset_j|
INTER = [[0] * NUM_SUBSETS for _ in range(NUM_SUBSETS)]
for i in range(NUM_SUBSETS):
    mi = SUBSET_MASKS[i]
    for j in range(i + 1, NUM_SUBSETS):
        mj = SUBSET_MASKS[j]
        inter_size = bin(mi & mj).count('1')
        INTER[i][j] = inter_size
        INTER[j][i] = inter_size

# Precompute bitsets for neighbors of each vertex v for each intersection size s
# B[v][s] is a bitmask of all u such that INTER[v][u] == s
# Note: INTER[v][v] is 4, but we only consider distinct sets.
# We only care about s in {0, 1, 2, 3}.
B = [[0] * 4 for _ in range(NUM_SUBSETS)]
for v in range(NUM_SUBSETS):
    for u in range(NUM_SUBSETS):
        if v == u:
            continue
        s = INTER[v][u]
        if s < 4:
            B[v][s] |= (1 << u)

# Precompute adjacency bitsets for each graph G_s
# E_s[u] is a bitmask of all w such that INTER[u][w] == s
# This is essentially the same as B[u][s]
# E_s = B[s] (transposed? No, B[v][s] is neighbors of v in G_s. So E_s[v] = B[v][s])
# We already have B.

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    best_family_size = 0
    best_family_indices = []
    
    # Number of restarts depends on speed.
    # We will run as many as possible.
    
    for attempt in range(1000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random order
        indices = list(range(NUM_SUBSETS))
        random.shuffle(indices)
        
        chosen = []
        chosen_mask = 0
        
        for idx in indices:
            v = idx
            
            # Check if v can be added to chosen
            can_add = True
            
            # For each s, check if G_s induced by (B[v][s] & chosen_mask) has an edge
            for s in range(4):
                # U is the set of neighbors of v in G_s that are already in chosen
                U = B[v][s] & chosen_mask
                if U == 0:
                    continue
                
                # Check if there is an edge in G_s within U
                # i.e., does there exist u, w in U such that INTER[u][w] == s?
                # This is equivalent to: does there exist u in U such that B[u][s] & U != 0?
                # Note: B[u][s] contains u's neighbors in G_s.
                
                temp = U
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    # Check if u has any neighbor in U within G_s
                    if B[u][s] & U:
                        can_add = False
                        break
                    temp ^= lsb
                if not can_add:
                    break
            
            if can_add:
                chosen.append(v)
                chosen_mask |= (1 << v)
        
        if len(chosen) > best_family_size:
            best_family_size = len(chosen)
            best_family_indices = chosen[:]
            
            # Early stop if we reach a very high value?
            # The max possible is 1820.
            # Theoretical bounds?
            # If we find > 1000, it's likely near optimal.
            # But let's keep running to be sure.
            if best_family_size > 1200:
                # We can break early if we are confident, but let's use time limit.
                pass

    # Convert to sorted lists
    result = [sorted(ALL_SUBSETS[i]) for i in best_family_indices]
    return result
