import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # 1. Precompute K(2,2,2) configurations
    # A configuration is defined by 3 disjoint pairs of points.
    # There are C(16,6) * 15 = 120,120 such configurations.
    # For each configuration, there are 8 transversal triples.
    # We need to find the conflict graph: two triples conflict if they are part of the same configuration?
    # No, the constraint is: "no three disjoint pairs have all eight transversal triples present".
    # This means we cannot have a set of 8 triples that form a K(2,2,2) transversal.
    # This is a 3-uniform hypergraph constraint (specifically, a 8-uniform constraint on the set of triples).
    # The standard "conflict graph" approach for Max Independent Set usually applies to pairwise conflicts.
    # Here, the conflict is 8-wise.
    # However, we can reformulate.
    # If we view each configuration (8 triples) as a "forbidden set", we are looking for a maximum subset of triples that does not contain any of these 8-sets.
    # This is the Maximum Independent Set in a hypergraph, or equivalently, a hitting set problem on the dual.
    
    # The directive suggests: "Formulate the problem as a maximum weight independent set in the conflict graph where vertices are triples and edges connect triples that complete a K(2,2,2) configuration".
    # This phrasing is slightly ambiguous. Usually, "edges connect triples that complete a K(2,2,2)" implies that if you pick both endpoints of an edge, you violate the constraint?
    # But a K(2,2,2) requires 8 triples.
    # Perhaps the directive implies a different structure or a relaxation.
    # Let's look at the "conflict" definition more closely.
    # If we pick 7 of the 8 transversal triples, we are "one away" from violation.
    # Adding the 8th is forbidden.
    # This is not a standard graph MIS problem.
    
    # However, we can use a Branch and Bound approach directly on the hypergraph constraints.
    # State: (Available Triples, Current Set)
    # Pruning: If the number of available triples + current size <= best found, prune.
    # Stronger Pruning: For any configuration, if we already have 7 of its 8 triples, we cannot add the 8th.
    
    # Let's implement a recursive Branch and Bound.
    # To make it efficient, we need good upper bounds.
    # A simple upper bound is the number of remaining available triples.
    # A better bound might involve counting how many "slots" are blocked.
    
    # Precomputation:
    # 1. Map triple to index.
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # 2. Identify all 8-sets (configurations).
    # We need to know, for each triple, which configurations it belongs to.
    # And for each configuration, which triples are in it.
    
    # Generate all 120,120 configurations.
    # Each config is a set of 8 triple indices.
    
    # Optimization: 
    # Instead of storing all 120k configs explicitly in memory as lists of 8 ints (which is fine, ~1M ints),
    # we can store the incidence.
    
    # Let's generate configs.
    # Points 0..15.
    # Choose 6 points.
    # Partition into 3 pairs.
    
    # We will store `configs` as a list of tuples of 8 triple indices.
    # And `triple_to_configs` as a list of lists of config indices.
    
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    
    # Precompute the 15 pairings of 6 items
    indices_6 = [0, 1, 2, 3, 4, 5]
    base_pairings = []
    seen_pairs = set()
    
    def gen_matchings(remaining):
        if not remaining:
            yield []
            return
        first = remaining[0]
        for i in range(1, len(remaining)):
            second = remaining[i]
            rest = remaining[1:i] + remaining[i+1:]
            for sub in gen_matchings(rest):
                yield [(first, second)] + sub

    for m in gen_matchings(indices_6):
        norm = tuple(sorted(tuple(sorted(p)) for p in m))
        if norm not in seen_pairs:
            seen_pairs.add(norm)
            base_pairings.append(norm)
            
    for pts in itertools.combinations(points, 6):
        for bp in base_pairings:
            actual_pairs = []
            for p in bp:
                actual_pairs.append((pts[p[0]], pts[p[1]]))
            
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in actual_pairs))
            
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tuple(sorted(tris)))
                
    num_configs = len(configs)
    
    # Incidence: triple -> list of config indices
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State for B&B:
    # We will use a bitset of available triples (560 bits -> 9 64-bit ints or Python ints).
    # Python big integers support bitwise operations efficiently.
    
    # Initial available set: all triples.
    all_triples_mask = (1 << num_triples) - 1
    
    best_count = 0
    best_set_mask = 0
    best_set_list = []
    
    # Heuristic for branching:
    # Pick the triple with the highest "impact" or most constraints?
    # Or just iterate in a fixed order for determinism in B&B?
    # Standard B&B: Pick a vertex v. Branch on "include v" and "exclude v".
    # To improve, we can use a greedy initial solution to set a high lower bound.
    
    # Let's run a fast greedy to get a good initial lower bound.
    # Greedy: Sort triples by degree (number of configs they are in) or just random?
    # The current program uses a line-based heuristic. Let's stick to a simple random/greedy mix for initialization.
    
    def run_greedy_init(seed):
        r = random.Random(seed)
        order = list(range(num_triples))
        r.shuffle(order)
        
        present = [False] * num_triples
        config_counts = [0] * num_configs
        current_set = []
        
        for t_idx in order:
            valid = True
            for cid in triple_to_configs[t_idx]:
                if config_counts[cid] == 7:
                    valid = False
                    break
            if valid:
                present[t_idx] = True
                current_set.append(t_idx)
                for cid in triple_to_configs[t_idx]:
                    config_counts[cid] += 1
                    
        return len(current_set), current_set

    # Run a few greedy inits to set the bar
    for s in [1, 2, 3, 4, 5]:
        cnt, lst = run_greedy_init(s)
        if cnt > best_count:
            best_count = cnt
            best_set_list = lst
            # Construct mask
            m = 0
            for t in lst:
                m |= (1 << t)
            best_set_mask = m

    # Branch and Bound
    # We need to track the "current available set" and "current chosen set".
    # To avoid copying large states, we can use recursion with backtracking on a stack or global state.
    # Given 560 vertices, depth is up to 560. Python recursion limit might be an issue, but we can increase it or use iterative deepening?
    # Actually, the depth of the search tree is the number of decisions.
    # We can use a recursive function that takes:
    #   available_mask: int
    #   current_count: int
    #   current_mask: int (for reconstruction if we beat best)
    
    # Upper Bound:
    # Simple: current_count + popcount(available_mask)
    # This is weak.
    # Better: For each config, if we have k triples from it in current_set, we can add at most (8-k) more from that config?
    # No, the constraint is on the *total* set.
    # If current_set has 7 triples from config C, we can add 0 more from C.
    # If current_set has 6, we can add 1 more from C, but adding it blocks the 8th.
    # This is complex to bound tightly without solving a subproblem.
    
    # Let's try a different pruning:
    # If there exists a config C such that all 8 triples of C are in `available_mask` OR `current_mask`?
    # No.
    
    # Let's stick to the simple bound first and see if it's fast enough with good branching.
    # Branching variable selection:
    # Choose a triple that appears in the most "critical" configs?
    # Or just the lowest index available?
    
    # To make B&B effective, we need to order the branching.
    # We will iterate through available triples.
    
    # Optimization: Precompute for each triple, the mask of triples that are "conflicting" in the sense that they share a config?
    # No, that's not a pairwise conflict.
    
    # Let's implement the B&B with a time check.
    
    sys.setrecursionlimit(10000)
    
    def popcount(x):
        return bin(x).count("1")
        
    def search(available_mask, current_count, current_mask):
        nonlocal best_count, best_set_mask, best_set_list
        
        if time.time() > deadline - 5:
            return
            
        # Prune 1: Upper bound
        if current_count + popcount(available_mask) <= best_count:
            return
            
        if available_mask == 0:
            if current_count > best_count:
                best_count = current_count
                best_set_mask = current_mask
                # Reconstruct list later
            return
            
        # Heuristic: Pick a triple to branch on.
        # Let's pick the one with the highest index? Or lowest?
        # Usually, picking a vertex with high degree in the conflict graph helps.
        # Here, "degree" could be the number of configs it belongs to.
        # Let's precompute config counts for each triple.
        
        # To avoid computing "best" triple every time (expensive), we can just pick the lowest set bit.
        # This is deterministic and fast.
        # v = lowest set bit in available_mask
        v = (available_mask & -available_mask).bit_length() - 1
        
        # Branch 1: Include v
        # Check if v is valid to add.
        # We need to know if adding v violates any config.
        # A config is violated if we already have 7 of its 8 triples in `current_mask` and v is the 8th?
        # No, we maintain `available_mask` such that if a config is "full" (7 in current), the 8th is removed from available?
        # This is the "forward checking" strategy.
        
        # Let's maintain `config_counts` in the recursion?
        # That requires passing an array or using a global array with backtracking.
        # Global array with backtracking is faster.
        
        # Global state for B&B:
        # config_counts_global = [0] * num_configs
        # We need to update it on entry/exit of recursion.
        
        # Let's refactor to use global config_counts.
        
    # Let's restart with a cleaner global state B&B implementation.
    
    config_counts = [0] * num_configs
    # We need to initialize config_counts based on an empty set (all 0).
    # The `available_mask` will dynamically remove triples that would violate constraints.
    
    # Helper: Check if triple t can be added given current config_counts.
    # It can be added if for all c in triple_to_configs[t], config_counts[c] < 7.
    
    # Helper: Update config_counts when adding/removing t.
    
    def can_add(t):
        for c in triple_to_configs[t]:
            if config_counts[c] >= 7:
                return False
        return True
        
    def add_triple(t):
        for c in triple_to_configs[t]:
            config_counts[c] += 1
            
    def remove_triple(t):
        for c in triple_to_configs[t]:
            config_counts[c] -= 1
            
    # Recursive B&B
    # available_mask: bits set for triples that are still candidates.
    # We assume that `available_mask` is already filtered such that any triple in it is *potentially* addable?
    # No, we filter dynamically.
    
    def bb_search(available_mask, current_count, current_mask):
        nonlocal best_count, best_set_mask, best_set_list
        
        if time.time() > deadline - 5:
            return
            
        if current_count + popcount(available_mask) <= best_count:
            return
            
        if available_mask == 0:
            if current_count > best_count:
                best_count = current_count
                best_set_mask = current_mask
            return
            
        # Select a vertex
        # Let's pick the vertex with the most constraints (highest degree in triple_to_configs)
        # among the available ones.
        # This is expensive to compute dynamically.
        # Let's precompute a static priority order and pick the first available one in that order.
        
        # Precomputed priority: sorted by len(triple_to_configs[i]) descending, then i.
        # static_priority = sorted(range(num_triples), key=lambda i: (-len(triple_to_configs[i]), i))
        
        # Find first available in static_priority
        v = -1
        for idx in static_priority:
            if (available_mask >> idx) & 1:
                v = idx
                break
                
        if v == -1:
            return # Should not happen if available_mask != 0
            
        # Branch 1: Include v
        if can_add(v):
            add_triple(v)
            new_available = available_mask & ~(1 << v)
            
            # Forward Checking: Remove from new_available any triple that can no longer be added
            # A triple u can no longer be added if there exists a config c in triple_to_configs[u]
            # such that config_counts[c] == 7 after adding v.
            # Note: Adding v increases config_counts for configs in triple_to_configs[v].
            # So we only need to check triples u that share a config with v?
            # Actually, if config_counts[c] becomes 7, then the 8th triple of c (if not v) is blocked.
            
            # Let's identify which configs reached 7.
            blocked_triples = 0
            for c in triple_to_configs[v]:
                if config_counts[c] == 7:
                    # The remaining triple in this config is blocked
                    for u in triples_in_config[c]:
                        if u != v and (available_mask >> u) & 1:
                            blocked_triples |= (1 << u)
                            
            new_available &= ~blocked_triples
            
            bb_search(new_available, current_count + 1, current_mask | (1 << v))
            
            remove_triple(v)
            
        # Branch 2: Exclude v
        new_available = available_mask & ~(1 << v)
        bb_search(new_available, current_count, current_mask)
        
    # Precompute static priority
    static_priority = sorted(range(num_triples), key=lambda i: (-len(triple_to_configs[i]), i))
    
    # Initial call
    # We can start with an empty set.
    # But we already have a good lower bound from greedy.
    # To speed up, we can initialize `available_mask` by removing triples that are "bad"?
    # No, let's just run B&B.
    
    # To avoid deep recursion overhead, we might need to be careful.
    # 560 depth is fine for Python if recursion limit is high.
    
    # One issue: The B&B might be too slow if the bound is weak.
    # Let's add a time check inside the loop if it's iterative?
    # The recursion itself is the loop.
    
    # Let's try running it.
    
    initial_available = all_triples_mask
    # We can prune the initial available set using the greedy solution?
    # No, the greedy solution is a subset.
    
    # Let's run B&B.
    # If it times out, we return the best found so far (from greedy or partial B&B).
    
    try:
        bb_search(initial_available, 0, 0)
    except RecursionError:
        pass # Should be handled by limit
        
    # Reconstruct best set from mask
    if best_set_mask:
        result = []
        for i in range(num_triples):
            if (best_set_mask >> i) & 1:
                result.append(all_triples[i])
        return result
    else:
        # Fallback to best greedy if B&B found nothing better (unlikely)
        return best_set_list

# Need to import sys for setrecursionlimit
import sys
sys.setrecursionlimit(10000)
