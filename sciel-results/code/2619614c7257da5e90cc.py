import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s limit for evaluation, so 115 is safe)

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices

    points = range(N)
    # We need to iterate over all sets of 6 points and all ways to partition them into 3 pairs.
    # The number of ways to partition 6 items into 3 pairs is 15.
    # Number of 6-point subsets is C(16, 6) = 8008.
    # Total configs ~ 8008 * 15 = 120,120.
    
    # To speed up precomputation, we can generate pairs directly.
    # A K(2,2,2) is defined by 3 disjoint pairs (a,b), (c,d), (e,f).
    # The 8 transversal triples are all combinations of one from each pair.
    
    # Let's generate all unique sets of 3 disjoint pairs.
    # There are C(16, 2) = 120 pairs.
    # We need to choose 3 disjoint pairs.
    # This is equivalent to choosing 6 points and partitioning them.
    
    # Efficient generation:
    # 1. Choose 6 points: C(16,6) = 8008
    # 2. For each set of 6 points, generate the 15 partitions into 3 pairs.
    
    all_configs_data = []
    
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        # 15 ways to partition 6 elements into 3 pairs
        pairs_list = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        
        for pair_set in pairs_list:
            # Normalize for uniqueness check (though within a specific 6-point set, partitions are unique)
            # But different 6-point sets might generate the same set of pairs? No, the pairs determine the 6 points.
            # So we just need to map the 8 triples.
            
            (a,b), (c,d), (e,f) = pair_set
            tris = []
            for x in (a,b):
                for y in (c,d):
                    for z in (e,f):
                        t = tuple(sorted((x,y,z)))
                        tris.append(triple_to_idx[t])
            
            # Store the 8 triple indices
            all_configs_data.append(tris)

    num_configs = len(all_configs_data)
    
    # For ILP, we need:
    # Variables x_i for each triple i (0..559)
    # Maximize sum(x_i)
    # Subject to: for each config k, sum(x_i for i in config_k) <= 7
    
    # Build constraint matrix A
    # A is sparse (num_configs x num_triples)
    # We can build it as a list of lists or use scipy.sparse
    
    from scipy.sparse import lil_matrix
    
    A = lil_matrix((num_configs, num_triples), dtype=np.int8)
    for k, tris in enumerate(all_configs_data):
        for t_idx in tris:
            A[k, t_idx] = 1
    
    A = A.tocsr()
    
    # Objective: maximize sum(x_i) -> minimize -sum(x_i)
    c = -np.ones(num_triples)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints: A @ x <= 7
    # LinearConstraint(A, lb=-np.inf, ub=7)
    constraints = LinearConstraint(A, lb=-np.inf, ub=7)
    
    # Run MILP
    # We need to be careful with the time limit.
    # scipy.optimize.milp does not have a direct time limit parameter in the standard interface,
    # but we can run it in a separate process or just hope it's fast enough?
    # Actually, `milp` uses HiGHS under the hood. HiGHS can be time-limited if we configure it, 
    # but the scipy wrapper doesn't expose that directly easily.
    
    # Alternative: Use a greedy initialization and then a very aggressive local search 
    # that uses the ILP relaxation to guide moves? 
    # Or just run the ILP and if it takes too long, fallback?
    # Given the small size (560 vars, 120k cons), HiGHS should solve this quickly 
    # or at least find a very good solution fast.
    
    # Let's try running it. If it hangs, we need a timeout mechanism.
    # Since we can't easily timeout scipy.optimize.milp inside the same thread without killing the process,
    # we will rely on the fact that this specific instance is likely solvable in seconds.
    # If not, we will implement a fallback local search.
    
    # To be safe, let's first do a fast greedy to get a baseline, then try ILP.
    
    # Fast Greedy Baseline
    present_greedy = np.zeros(num_triples, dtype=bool)
    config_counts_greedy = np.zeros(num_configs, dtype=np.int8)
    
    # Precompute triple_to_configs for local search speed
    triple_to_configs = [[] for _ in range(num_triples)]
    for k, tris in enumerate(all_configs_data):
        for t_idx in tris:
            triple_to_configs[t_idx].append(k)
            
    def add_triple_greedy(t_idx):
        present_greedy[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts_greedy[cid] += 1
            
    def remove_triple_greedy(t_idx):
        present_greedy[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts_greedy[cid] -= 1
            
    def is_valid_addition_greedy(t_idx):
        if present_greedy[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts_greedy[cid] == 7:
                return False
        return True

    # Run Greedy
    order = list(range(num_triples))
    random.seed(42)
    random.shuffle(order)
    for t_idx in order:
        if is_valid_addition_greedy(t_idx):
            add_triple_greedy(t_idx)
            
    greedy_score = np.sum(present_greedy)
    
    # Now try ILP
    # We will run ILP with a "best effort" approach.
    # If it returns a solution, we use it.
    
    try:
        # Set a timeout by checking time before and after? No, it blocks.
        # Let's assume it's fast. If it's not, the program might timeout.
        # To mitigate risk, we can run a limited local search instead if ILP is risky.
        # However, ILP is the "different strategy".
        
        # Let's try to run it.
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_triples))
        
        if res.status == 0: # Optimal
            x = res.x
            # x might be fractional if integrality wasn't enforced perfectly? 
            # integrality=1 means binary.
            # Round to nearest integer
            x_bin = (x > 0.5).astype(np.int8)
            ilp_score = np.sum(x_bin)
            
            if ilp_score > greedy_score:
                present_final = x_bin.astype(bool)
            else:
                present_final = present_greedy.copy()
        else:
            present_final = present_greedy.copy()
            
    except Exception as e:
        # If ILP fails or times out (though it doesn't timeout naturally, it might crash or take too long)
        # We fall back to greedy
        present_final = present_greedy.copy()
        
    # Final Polish: Local Search on the best state found
    # We will run a quick local search to see if we can improve the ILP/Greedy result.
    # This ensures we don't lose points due to numerical issues or suboptimal ILP settings.
    
    present = present_final.copy()
    config_counts = np.zeros(num_configs, dtype=np.int8)
    for i in range(num_triples):
        if present[i]:
            for cid in triple_to_configs[i]:
                config_counts[cid] += 1
                
    current_score = np.sum(present)
    
    # Simple Local Search: Try to add any missing triple.
    # Then try swaps: remove 1, add 2? No, we want to maximize.
    # The ILP solution should be locally optimal w.r.t single additions.
    # Let's do a few rounds of "try to add" just in case.
    
    for _ in range(10):
        added_any = False
        for t_idx in range(num_triples):
            if not present[t_idx] and is_valid_addition_greedy(t_idx):
                # Wait, is_valid_addition_greedy uses the global greedy state.
                # We need a local version or update the global one.
                # Let's just use the local present/config_counts.
                pass
        
        # Let's implement a local check
        def is_valid_local(t_idx):
            if present[t_idx]:
                return False
            for cid in triple_to_configs[t_idx]:
                if config_counts[cid] == 7:
                    return False
            return True

        def add_local(t_idx):
            present[t_idx] = True
            for cid in triple_to_configs[t_idx]:
                config_counts[cid] += 1

        def remove_local(t_idx):
            present[t_idx] = False
            for cid in triple_to_configs[t_idx]:
                config_counts[cid] -= 1

        for t_idx in range(num_triples):
            if not present[t_idx] and is_valid_local(t_idx):
                add_local(t_idx)
                current_score += 1
                added_any = True
                
        if not added_any:
            break
            
    # Convert back to list of triples
    result = [all_triples[i] for i in range(num_triples) if present[i]]
    
    return result
