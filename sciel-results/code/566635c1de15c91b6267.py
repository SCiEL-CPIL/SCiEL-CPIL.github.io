import time
import random
from itertools import combinations

def construct():
    """
    Construct a simple graph on n vertices with no induced regular subgraph
    on 6 or more vertices.
    """
    start_time = time.time()
    # Leave some time for final processing and safety
    deadline = start_time + 145 

    # We aim for n=20, which is known to be possible.
    # We will try to find a valid graph on 20 vertices.
    
    best_result = None

    for n in [20, 19]:
        if time.time() > deadline - 5:
            break
            
        # Try multiple random seeds/restarts for this n
        for seed in range(100):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            
            # Initialize random graph
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            # Local search to remove violations
            # We iterate until no violations are found or time runs out
            iter_count = 0
            max_iters = 5000 # Hard cap on iterations per restart to avoid infinite loops
            
            while time.time() < deadline and iter_count < max_iters:
                iter_count += 1
                viol = find_violation(adj_bits, n)
                
                if viol is None:
                    # Success!
                    # Extract edges
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj_bits[i] >> j) & 1:
                                edges.append([i, j])
                    
                    # Verify one last time just in case
                    if is_valid(adj_bits, n):
                        return {'n': n, 'edges': edges}
                    # If verification fails (should not happen), treat as failure and continue
                    break
                
                # Found a violation 'viol' (list of indices)
                # Pick a random edge within 'viol' to flip
                # We pick two random vertices from the violation set
                u, v = random.sample(viol, 2)
                
                # Flip edge (u, v)
                if (adj_bits[u] >> v) & 1:
                    # Edge exists, remove it
                    adj_bits[u] ^= (1 << v)
                    adj_bits[v] ^= (1 << u)
                else:
                    # Edge doesn't exist, add it
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)

    # If we failed to find a valid graph on 20 or 19, fallback to a smaller valid graph
    # The empty graph on 5 vertices is always valid.
    # Or we can try a fixed small graph.
    return {'n': 5, 'edges': []}

def find_violation(adj_bits, n):
    """
    Find any induced regular subgraph of size k >= 6.
    Returns the list of indices if found, else None.
    Checks smaller k first.
    """
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            # Create mask for subset
            mask = 0
            for idx in subset:
                mask |= (1 << idx)
            
            # Calculate degrees of vertices in subset
            # We only need to check if all degrees are equal
            # Optimization: compute first degree, then compare others
            first_deg = None
            is_regular = True
            
            for idx in subset:
                # Count neighbors of idx within the subset
                # adj_bits[idx] contains bits for neighbors
                # We mask with 'mask' to keep only neighbors in subset
                common = adj_bits[idx] & mask
                deg = common.bit_count()
                
                if first_deg is None:
                    first_deg = deg
                elif deg != first_deg:
                    is_regular = False
                    break
            
            if is_regular:
                return list(subset)
                
    return None

def is_valid(adj_bits, n):
    """
    Verify that no induced subgraph of size >= 6 is regular.
    """
    return find_violation(adj_bits, n) is None

