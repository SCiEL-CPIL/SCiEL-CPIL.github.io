import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Generate all triples
    all_triples = []
    triple_index = {}
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                t = (a, b, c)
                triple_index[t] = len(all_triples)
                all_triples.append(t)
    
    n_triples = len(all_triples)
    # Bitset representation: bit i is 1 if triple i is present
    present = (1 << n_triples) - 1
    
    # Precompute for each 6-subset, the 90 partitions into 3 pairs
    # and the 8 transversal triples for each partition
    six_subsets = list(combinations(range(16), 6))
    k222_configs = []
    for pts in six_subsets:
        # Generate all ways to partition 6 points into 3 pairs
        # (a1,a2), (b1,b2), (c1,c2)
        # We can fix a1=pts[0], a2=pts[1], then choose b1,b2 from remaining 4, then c1,c2 are the last 2
        # But we need to consider all 90 distinct partitions.
        # Actually, the number of ways to partition 6 distinct items into 3 unordered pairs is 15.
        # But the pairs are labeled (A, B, C) in K(2,2,2), so we need to consider labeled partitions.
        # The number of labeled partitions is 6! / (2!^3) = 90.
        # We can generate them by:
        # Choose 2 for A: C(6,2)=15
        # Choose 2 for B from remaining 4: C(4,2)=6
        # C is the last 2: 1
        # Total 15*6*1=90.
        # For each, the 8 transversal triples are {a_i, b_j, c_k} for i,j,k in {0,1}.
        
        # To avoid duplicates in the 6-subset iteration, we just iterate over all 90 labeled partitions.
        for a_pair in combinations(pts, 2):
            a1, a2 = a_pair
            rem1 = [x for x in pts if x not in a_pair]
            for b_pair in combinations(rem1, 2):
                b1, b2 = b_pair
                c_pair = tuple(x for x in rem1 if x not in b_pair)
                c1, c2 = c_pair
                # The 8 transversal triples
                transversals = []
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            t = tuple(sorted((a_pair[i], b_pair[j], c_pair[k])))
                            transversals.append(t)
                k222_configs.append(transversals)
    
    # Map each triple to its index
    # present is a bitset
    
    # Greedy removal
    # We need to find all K(2,2,2) in the current set
    # For each config, check if all 8 triples are present
    # Count how many K(2,2,2) each triple is in
    
    while True:
        if time.time() > deadline:
            break
        
        # Find all K(2,2,2)
        conflict_count = [0] * n_triples
        found_conflict = False
        for transversals in k222_configs:
            # Check if all 8 are present
            all_present = True
            for t in transversals:
                idx = triple_index[t]
                if not (present >> idx) & 1:
                    all_present = False
                    break
            if all_present:
                found_conflict = True
                for t in transversals:
                    idx = triple_index[t]
                    conflict_count[idx] += 1
        
        if not found_conflict:
            break
        
        # Find the triple with the highest conflict count
        max_count = -1
        max_idx = -1
        for i in range(n_triples):
            if conflict_count[i] > max_count:
                max_count = conflict_count[i]
                max_idx = i
        
        # Remove the triple
        present &= ~(1 << max_idx)
    
    # Convert bitset to list of triples
    result = []
    for i in range(n_triples):
        if (present >> i) & 1:
            result.append(list(all_triples[i]))
    
    return result
