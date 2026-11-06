import time
import random
from itertools import combinations

# Precomputation
N = 16
ALL_TRIPLES = list(combinations(range(N), 3))
NUM_TRIPLES = len(ALL_TRIPLES) # 560

# Map triple to index
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Generate all K(2,2,2) configurations
# A K(2,2,2) is defined by 3 disjoint pairs.
# We iterate over all sets of 6 vertices.
# For each set of 6, there are 15 ways to partition into 3 pairs.

def generate_k222s():
    k222s = []
    # Iterate over all combinations of 6 vertices
    for verts in combinations(range(N), 6):
        v = list(verts)
        # Pair v0 with v1..v5 (5 choices)
        for i in range(1, 6):
            pair0 = (v[0], v[i])
            remaining = [v[j] for j in range(6) if j != 0 and j != i]
            # Pair remaining 4 into 2 pairs (3 choices)
            # remaining = [r0, r1, r2, r3]
            # Choices: (r0,r1)(r2,r3), (r0,r2)(r1,r3), (r0,r3)(r1,r2)
            
            # Option 1
            pair1 = (remaining[0], remaining[1])
            pair2 = (remaining[2], remaining[3])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
            
            # Option 2
            pair1 = (remaining[0], remaining[2])
            pair2 = (remaining[1], remaining[3])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
            
            # Option 3
            pair1 = (remaining[0], remaining[3])
            pair2 = (remaining[1], remaining[2])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
    return k222s

K222S = generate_k222s()
NUM_K222 = len(K222S)

# Map each edge to the list of K222 indices it belongs to
edge_to_k222s = [[] for _ in range(NUM_TRIPLES)]
for k_idx, edges in enumerate(K222S):
    for e_idx in edges:
        edge_to_k222s[e_idx].append(k_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    best_count = 0
    best_system = []
    
    # Precompute a list of triples for random access
    triple_list = ALL_TRIPLES
    
    # We will run multiple iterations of:
    # 1. Random Greedy Initialization
    # 2. Local Search (Add, Remove, Swap)
    
    while time.time() < deadline:
        # 1. Greedy Initialization
        # Shuffle triples
        shuffled_indices = list(range(NUM_TRIPLES))
        random.shuffle(shuffled_indices)
        
        present = [False] * NUM_TRIPLES
        counts = [0] * NUM_K222
        current_count = 0
        
        for idx in shuffled_indices:
            # Check if adding idx is valid
            valid = True
            for k_idx in edge_to_k222s[idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                present[idx] = True
                current_count += 1
                for k_idx in edge_to_k222s[idx]:
                    counts[k_idx] += 1
        
        if current_count > best_count:
            best_count = current_count
            best_system = [triple_list[i] for i in range(NUM_TRIPLES) if present[i]]
        
        # 2. Local Search
        # We maintain 'present' and 'counts'.
        # We will perform swaps and adds.
        
        # Create a list of absent triples for efficient random selection
        # Updating this list dynamically is expensive, so we might just pick random indices and check.
        # Or we can maintain a set of absent indices.
        absent_set = set(i for i in range(NUM_TRIPLES) if not present[i])
        
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap Move: Pick an absent triple, try to make room for it
                if not absent_set:
                    break
                
                new_idx = random.choice(list(absent_set))
                
                # Check if new_idx is blocked
                blocking_k222s = []
                for k_idx in edge_to_k222s[new_idx]:
                    if counts[k_idx] == 7:
                        blocking_k222s.append(k_idx)
                
                if not blocking_k222s:
                    # Not blocked, just add it
                    present[new_idx] = True
                    absent_set.remove(new_idx)
                    current_count += 1
                    for k_idx in edge_to_k222s[new_idx]:
                        counts[k_idx] += 1
                    no_improve_count = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_system = [triple_list[i] for i in range(NUM_TRIPLES) if present[i]]
                else:
                    # It is blocked. We need to remove one edge from one of the blocking K222s.
                    # Pick a random blocking K222
                    k_idx = random.choice(blocking_k222s)
                    # The edges in this K222 that are present are the ones we can remove.
                    # There are exactly 7 present edges in this K222.
                    # We pick one at random to remove.
                    edges_in_k222 = K222S[k_idx]
                    # Filter to present edges
                    present_edges_in_k222 = [e for e in edges_in_k222 if present[e]]
                    
                    if present_edges_in_k222:
                        rem_idx = random.choice(present_edges_in_k222)
                        
                        # Remove rem_idx
                        present[rem_idx] = False
                        absent_set.add(rem_idx)
                        current_count -= 1
                        for k_idx_2 in edge_to_k222s[rem_idx]:
                            counts[k_idx_2] -= 1
                        
                        # Try to add new_idx
                        valid_add = True
                        for k_idx_3 in edge_to_k222s[new_idx]:
                            if counts[k_idx_3] == 7:
                                valid_add = False
                                break
                        
                        if valid_add:
                            # Add new_idx
                            present[new_idx] = True
                            absent_set.remove(new_idx)
                            current_count += 1
                            for k_idx_4 in edge_to_k222s[new_idx]:
                                counts[k_idx_4] += 1
                            # Net change 0, but structure improved
                            no_improve_count = 0
                        else:
                            # Revert removal
                            present[rem_idx] = True
                            absent_set.remove(rem_idx)
                            current_count += 1
                            for k_idx_5 in edge_to_k222s[rem_idx]:
                                counts[k_idx_5] += 1
                            no_improve_count += 1
                            
            elif move_type < 0.7:
                # Add Move: Pick an absent triple, add if possible
                if not absent_set:
                    break
                new_idx = random.choice(list(absent_set))
                valid = True
                for k_idx in edge_to_k222s[new_idx]:
                    if counts[k_idx] == 7:
                        valid = False
                        break
                if valid:
                    present[new_idx] = True
                    absent_set.remove(new_idx)
                    current_count += 1
                    for k_idx in edge_to_k222s[new_idx]:
                        counts[k_idx] += 1
                    no_improve_count = 0
                    if current_count > best_count:
                        best_count = current_count
                        best_system = [triple_list[i] for i in range(NUM_TRIPLES) if present[i]]
                else:
                    no_improve_count += 1
                    
            else:
                # Remove Move: Remove a random present triple (to allow future adds)
                # Only useful if we are stuck, but generally reduces count.
                # We only do this if we have been stuck for a while?
                # Or just rarely.
                if no_improve_count > 100:
                    present_indices = [i for i in range(NUM_TRIPLES) if present[i]]
                    if present_indices:
                        rem_idx = random.choice(present_indices)
                        present[rem_idx] = False
                        absent_set.add(rem_idx)
                        current_count -= 1
                        for k_idx in edge_to_k222s[rem_idx]:
                            counts[k_idx] -= 1
                        no_improve_count = 0 # Reset as we changed state significantly
        
    return best_system
