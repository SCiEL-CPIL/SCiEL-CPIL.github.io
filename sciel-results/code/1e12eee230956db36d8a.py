import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a dual-guided greedy heuristic
    formulated as a Minimum Weight Vertex Cover on the bipartite graph between
    t-subsets and k-blocks.
    """
    # Generate all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Track which t-subsets are uncovered
    uncovered = [True] * n_tsubsets
    uncovered_count = n_tsubsets
    
    # Dual weights: initially 1 for all
    dual = [1.0] * n_tsubsets
    
    # Precompute for each point, the list of t-subsets containing it
    point_to_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubsets):
        for x in ts:
            point_to_tsubs[x].append(i)
    
    blocks = []
    
    # For efficiency, we'll maintain a set of uncovered t-subset indices
    uncovered_set = set(range(n_tsubsets))
    
    while uncovered_set:
        # Find the uncovered t-subset with the highest dual weight
        # Use a priority approach: pick the one with max dual among uncovered
        best_tsub_idx = max(uncovered_set, key=lambda i: dual[i])
        best_dual = dual[best_tsub_idx]
        
        # Start the block with the points from this t-subset
        block = list(all_tsubsets[best_tsub_idx])
        block_set = set(block)
        
        # Greedily add points to maximize the weighted coverage of uncovered t-subsets
        while len(block) < k:
            best_point = -1
            best_gain = -1.0
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Compute the gain: sum of dual weights of uncovered t-subsets
                # that contain all of block and x, i.e., S \ {x} ⊆ block and x ∈ S
                gain = 0.0
                # For each (t-1)-subset R of block, S = R ∪ {x}
                for R in itertools.combinations(block, t - 1):
                    S = tuple(sorted(R + (x,)))
                    idx = tsub_to_idx.get(S, -1)
                    if idx >= 0 and idx in uncovered_set:
                        gain += dual[idx]
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point < 0 or best_gain <= 0:
                # No point improves coverage; add a random point
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_point = random.choice(remaining)
                else:
                    break
            
            block.append(best_point)
            block_set.add(best_point)
        
        # Sort the block
        block.sort()
        blocks.append(block)
        
        # Update uncovered status and dual weights
        newly_covered = []
        for i in list(uncovered_set):
            ts = all_tsubsets[i]
            if all(x in block_set for x in ts):
                uncovered_set.remove(i)
                uncovered_count -= 1
                newly_covered.append(i)
        
        # Adjust dual weights: increase dual for uncovered t-subsets that share
        # few points with the newly covered ones (harder to cover)
        for i in list(uncovered_set):
            ts = all_tsubsets[i]
            # Count overlap with the block
            overlap = sum(1 for x in ts if x in block_set)
            # If low overlap, it's harder to cover, increase dual
            if overlap < t:
                dual[i] *= 1.1
    
    return blocks
