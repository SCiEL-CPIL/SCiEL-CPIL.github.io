import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precomputation
    # Map each t-subset to an index
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}
    
    # covered: boolean numpy array of size n_subsets
    covered = np.zeros(n_subsets, dtype=bool)
    
    # Precompute triple indices for fast gain calculation
    # triple_idx[x][u][v] = index of subset (x, u, v) if x < u < v, else -1
    # We need to handle any order, so we sort.
    # Let's create a 3D array of shape (v, v, v)
    triple_idx = np.full((v, v, v), -1, dtype=np.int32)
    for i, s in enumerate(all_subsets):
        a, b, c = s
        triple_idx[a][b][c] = i
        triple_idx[a][c][b] = i
        triple_idx[b][a][c] = i
        triple_idx[b][c][a] = i
        triple_idx[c][a][b] = i
        triple_idx[c][b][a] = i

    blocks = []
    
    def get_gain(x, B_list):
        """
        Compute the number of new t-subsets covered by adding x to B_list.
        B_list is a list of points already in the block.
        """
        if len(B_list) < t - 1:
            return 0
        
        # We need to find indices of subsets {x, u, v} where u, v in B_list
        # and the subset is not yet covered.
        
        # Create a list of pairs (u, v) from B_list
        # Since B_list is small (max 6), we can just iterate
        # But we want to use numpy for the coverage check.
        
        # Get indices for all triples involving x and pairs from B_list
        # We can construct an array of indices
        indices = []
        for i in range(len(B_list)):
            for j in range(i+1, len(B_list)):
                u = B_list[i]
                v = B_list[j]
                idx = triple_idx[x][u][v]
                if idx != -1:
                    indices.append(idx)
        
        if not indices:
            return 0
            
        idx_arr = np.array(indices, dtype=np.int32)
        # Check which are not covered
        # covered[idx_arr] gives a boolean array
        # Count True values
        return np.sum(~covered[idx_arr])

    def build_block_from(seed_points, current_covered):
        """
        Grow a block from seed_points to size k using greedy max-gain.
        current_covered is a numpy boolean array.
        """
        B = list(seed_points)
        B_set = set(B)
        available = [x for x in range(v) if x not in B_set]
        
        while len(B) < k:
            best_x = -1
            best_gain = -1
            
            for x in available:
                g = get_gain(x, B)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                break
            
            B.append(best_x)
            B_set.add(best_x)
            available.remove(best_x)
            
        return B

    def update_coverage(block):
        """
        Update the covered array with the triples in the block.
        """
        for s in itertools.combinations(sorted(block), t):
            idx = subset_to_index[s]
            covered[idx] = True

    while True:
        if time.time() > deadline:
            break
            
        # Check if all covered
        if np.all(covered):
            break
            
        # Find uncovered subset indices
        # To limit search, we can sample or take the first few
        # Let's take the first 100 uncovered indices as seeds
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Sample seeds if too many
        if len(uncovered_indices) > 50:
            # Random sample for diversity, or just first 50?
            # Random sample might find better blocks
            np.random.seed(int(time.time() * 1000) % 10000)
            sampled = np.random.choice(uncovered_indices, 50, replace=False)
        else:
            sampled = uncovered_indices
            
        best_block = None
        best_gain_total = -1
        
        for idx in sampled:
            if time.time() > deadline:
                break
            seed = all_subsets[idx]
            B = build_block_from(list(seed), covered)
            
            # Calculate total new coverage of B
            # We need to count how many triples in B are NOT in covered
            new_count = 0
            for s in itertools.combinations(sorted(B), t):
                s_idx = subset_to_index[s]
                if not covered[s_idx]:
                    new_count += 1
            
            if new_count > best_gain_total:
                best_gain_total = new_count
                best_block = B
                
        if best_block is None:
            break
            
        bl = sorted(best_block)
        update_coverage(bl)
        blocks.append(bl)
        
    return blocks
