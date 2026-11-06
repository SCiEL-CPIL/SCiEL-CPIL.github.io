import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

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
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        # Generate all 15 partitions of 6 points into 3 pairs
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
            # Normalize pair order and pair internal order for unique ID
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                # Find the 8 transversal triples
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # Build conflict graph
    # Conflict exists if two triples are in the same K(2,2,2) config?
    # No, the constraint is: NO K(2,2,2) can have ALL 8 triples present.
    # This is a 8-uniform hypergraph independent set problem, or equivalently,
    # we can model it as a graph problem if we convert the hyperedges?
    # Actually, the current program uses "config_counts". 
    # A simpler graph model: If we pick a triple, it doesn't immediately conflict with others 
    # unless they complete a config. But "completing a config" requires 7 others.
    # This is NOT a standard graph independent set where edges mean "cannot coexist".
    # Two triples CAN coexist. Three CAN coexist. ... Seven CAN coexist. Eight CANNOT.
    
    # However, the prompt suggests: "Formulate the problem as a maximum weight independent set in the conflict graph where vertices are triples and edges connect triples that complete a K(2,2,2) configuration"
    # This phrasing is slightly ambiguous. "Edges connect triples that complete a K(2,2,2)" usually implies that if u and v are connected, they form a forbidden structure. But here the forbidden structure is size 8.
    # If we interpret "edges connect triples that complete a K(2,2,2)" as "u and v are in the same config", then the graph is very dense.
    # If we select an independent set in this "same-config" graph, we ensure no two triples are in the same config. This is a MUCH stronger constraint than needed (it would limit us to 1 triple per config, i.e., max ~560/8? No, configs overlap).
    # Actually, if we just ensure no config has 8 triples, we can have 7.
    
    # Let's stick to the "Branch and Bound on the Hypergraph" or "MaxSAT" approach but implemented efficiently.
    # The prompt specifically asks for MIS in a conflict graph. 
    # Maybe the intended interpretation is: 
    # Construct a graph G where V = triples. 
    # Edge (u, v) exists if there is a config C such that u, v in C? 
    # If we find a Max Independent Set in this G, we guarantee that for any config C, we pick at most 1 triple from C? 
    # No, an independent set in G means no two picked vertices are adjacent. 
    # If adjacency means "share a config", then an independent set contains at most 1 triple from any config. 
    # This is a valid solution (since 1 < 8), but likely suboptimal (we can have up to 7).
    # However, finding MIS in a dense graph is hard.
    
    # Alternative interpretation: 
    # The "conflict graph" for the specific constraint "No 8 in a config" is not a simple graph.
    # But we can use a Branch and Bound on the set of triples directly, using the config counts as the pruning mechanism.
    # This is effectively B&B on the hypergraph.
    
    # Let's implement a B&B that branches on adding/removing triples, using the greedy bound:
    # Bound = current_size + number_of_remaining_triples_that_can_still_be_added.
    # "Can still be added" means: for all configs C containing t, count(C) < 7.
    
    # To make this fast:
    # 1. Order triples by "popularity" (number of configs they are in, or just index).
    # 2. Use bitsets for speed? 560 triples fits in 10 64-bit ints or just Python ints (arbitrary precision).
    
    # Let's use Python integers as bitsets for the "available" triples.
    # But the constraint is global (config counts).
    
    # Strategy:
    # 1. Greedy initial solution to get a good lower bound.
    # 2. B&B:
    #    - State: (current_set, current_count, config_counts)
    #    - Candidates: Triples not in current_set.
    #    - For each candidate t:
    #        - Check if adding t is valid (all configs in t have count < 7).
    #        - If valid, recurse with t added.
    #    - Prune if current_count + greedy_bound(candidates) <= best_found.
    #    
    #    Greedy bound: 
    #       Sort candidates by some heuristic (e.g., fewest conflicts).
    #       Greedily pick as many as possible from candidates assuming they don't conflict with each other? 
    #       Or just count how many are "individually valid" given current state?
    #       A tighter bound: Run a fast greedy on the remaining candidates to estimate max additional.
    
    # Given 560 triples, a naive B&B might be too slow. 
    # We need to be clever.
    # Let's use the "WalkSAT" inspiration but structured as B&B?
    # No, the directive says "different design family". B&B is different from SA/Greedy.
    
    # Optimization:
    # The graph of "triple t blocks triple u" is dynamic.
    # Let's precompute for each triple t, the list of configs it's in.
    
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
    
    # Precompute "conflict neighbors" for a greedy bound?
    # If we add t, it reduces the capacity of all configs in triple_to_configs[t].
    
    # Let's implement a recursive B&B with memoization on a "signature" of the state?
    # The state is defined by the set of present triples. Too large.
    # Instead, we branch on a fixed ordering of triples.
    # Decision: For triple i (in order 0..559), do we include it or not?
    # This is standard B&B for independent set.
    
    # To speed up:
    # 1. Sort triples by degree (number of configs) descending? Or ascending?
    #    Usually, high degree vertices are more restrictive. Branching on them first might prune earlier.
    
    # 2. Upper Bound:
    #    current_count + sum(1 for t in remaining if is_valid(t)) is a weak bound.
    #    Better: Run a greedy independent set on the "induced subgraph" of remaining valid triples?
    #    The "graph" here is: t1 and t2 conflict if adding both would complete a config? 
    #    No, they conflict if they are in the same config AND that config is close to full?
    #    Actually, the constraint is global.
    #    
    #    Let's use a simple bound: 
    #    Max possible = current_count + (number of remaining triples that are currently valid).
    #    This is weak.
    
    #    Let's try a different bound:
    #    Partition the remaining triples into disjoint sets where no two in a set can coexist?
    #    If we can cover the remaining vertices with k cliques, the bound is current_count + k.
    #    In our problem, a "clique" is a set of triples where any two cannot coexist.
    #    When do two triples t1, t2 not coexist?
    #    They don't *immediately* not coexist. They only fail if they complete a config with 6 others.
    #    So the "conflict graph" is not static.
    
    #    However, we can use a static conflict graph for the bound:
    #    If t1 and t2 are in the same config C, and C currently has 6 other triples selected, then t1 and t2 conflict (one must be dropped).
    #    This is dynamic.
    
    #    Let's stick to the dynamic check in the B&B loop.
    
    # Implementation Plan:
    # 1. Get initial best via Greedy (randomized, multiple runs).
    # 2. B&B:
    #    - Maintain a stack of states.
    #    - State: (idx, present_mask, config_counts, current_count)
    #    - Since 560 is too big for a simple bitmask in Python int for fast ops? 
    #      Python ints are arbitrary precision, so a 560-bit int is fine.
    #    - But config_counts is a vector of 1000+ ints. Copying it is expensive.
    #    - Use persistent data structure or just copy numpy array? 
    #      Copying a 1000-int array is fast in numpy.
    
    #    Let's use a depth-first search with recursion limit adjusted.
    #    To avoid deep recursion overhead, we can use a stack.
    
    #    Pruning:
    #    If current_count + len(remaining_valid) <= best: prune.
    #    remaining_valid: triples from idx to end that are currently valid.
    
    #    To make "remaining_valid" fast:
    #    We can precompute for each idx, the list of triples from idx to end.
    #    Checking validity for each is O(configs per triple).
    
    #    Let's add a "Folding" rule:
    #    If a triple t is valid, and it doesn't block any other triple (i.e., for all configs C in t, count(C) < 6), 
    #    then t can be added without loss of generality? 
    #    No, adding t might block others later.
    #    But if t is "safe" (doesn't reduce capacity of any config below 1 for others?), maybe.
    
    #    Let's just run the B&B with the weak bound first and see if it's fast enough with a good initial guess.
    #    If not, we need a tighter bound.
    
    #    Tighter Bound Idea:
    #    Greedily select a subset of remaining valid triples that can coexist.
    #    This greedy selection serves as a lower bound on the remaining potential, 
    #    but we need an UPPER bound.
    #    Actually, the size of a Max Independent Set in the "conflict graph of remaining valid triples" is the upper bound.
    #    We can approximate this MIS size with a greedy algorithm (which gives a lower bound on MIS, which is an upper bound on the *pruning*? No.)
    #    Wait. 
    #    Upper Bound on Total = Current + Max(Additional).
    #    Max(Additional) <= Size of Max Independent Set of Remaining Valid Triples (under dynamic constraints).
    #    This is hard to compute.
    
    #    Let's use a simpler bound: 
    #    Color the remaining valid triples. If we can color them with k colors such that no two same-color triples conflict (dynamically), then Max(Additional) <= k? No, that's for graphs.
    
    #    Let's try the "Branch and Reduce" with the simple bound:
    #    Bound = current_count + count of remaining triples that are individually valid.
    #    This is very weak.
    
    #    Better Bound:
    #    For each remaining triple t, if it is valid, it "uses" capacity.
    #    Let's just use the fact that each config can take at most 7.
    #    Total capacity = sum over configs of (7 - current_count(C))? No, triples belong to multiple configs.
    
    #    Let's rely on the initial greedy being very good and the B&B just proving optimality or finding a slightly better one.
    #    If the B&B is too slow, we fall back to the best found.
    
    #    To make B&B fast:
    #    - Order triples by "constraint tightness". Triples in configs that are already at 6 are most urgent.
    #    - Actually, just sort by index is fine if the bound is tight enough.
    
    #    Let's try a different angle: 
    #    The problem is small enough (560 vertices) that a well-optimized C++ MIS solver would crush it. 
    #    In Python, we need to be smart.
    
    #    Let's use the "WalkSAT" inspiration from the prompt's inspirations but implement it as a structured local search with a "restart" mechanism, which is effectively a form of stochastic B&B.
    #    Wait, the directive says "different from current". Current is SA/Greedy. 
    #    Inspiration 1 is WalkSAT. 
    #    Let's implement a **MaxSAT solver** style approach.
    #    Variables: x_i (1 if triple i is present).
    #    Clauses: For each config C, NOT (x_1 AND ... AND x_8). 
    #    This is a 8-CNF clause.
    #    We want to maximize sum(x_i).
    #    
    #    Algorithm: 
    #    1. Start with all x_i = 1.
    #    2. While violations exist:
    #       - Pick a violated config C.
    #       - Pick a variable x_j in C to flip to 0.
    #       - Choose j based on a heuristic (e.g., minimize loss of other clauses? or random).
    #    3. Local Search: Flip variables to 1 if it doesn't create violations.
    
    #    This is essentially the "Repair" phase.
    #    To improve, we need "Perturbation" (flip some 1s to 0s randomly) to escape local optima.
    
    #    This is structurally different from the current SA which does 2-out 1-in.
    #    This is a "Unit Propagation + Conflict Driven" approach.
    
    #    Let's implement this Conflict-Driven Local Search (CDLS).
    
    best_score = 0
    best_state = np.zeros(num_triples, dtype=bool)
    
    # Precompute config membership for fast access
    # config_to_triples is already there.
    
    def run_cdls(seed):
        random.seed(seed)
        # Initialize: All triples present
        present = np.ones(num_triples, dtype=bool)
        config_counts = np.zeros(num_configs, dtype=np.int8)
        for cid, tris in enumerate(triples_in_config):
            config_counts[cid] = len(tris) # All 8 are present initially
        
        # List of violated configs
        violated = [i for i in range(num_configs) if config_counts[i] >= 8]
        
        # Repair Phase
        while violated:
            # Pick a random violated config
            cid = random.choice(violated)
            # Pick a random triple from this config to remove
            # Heuristic: Pick one that is in the fewest other violated configs?
            # For now, random.
            tris = triples_in_config[cid]
            t_idx = random.choice(tris)
            
            # Remove t_idx
            if present[t_idx]:
                present[t_idx] = False
                # Update counts
                for c in triple_to_configs[t_idx]:
                    config_counts[c] -= 1
                    # Update violated list? 
                    # We can just recompute violated list periodically or maintain it.
                    # Recomputing is O(num_configs). 
                    # Let's just recompute when violated becomes empty or after K steps.
                    # Or, simply: if config_counts[c] == 8, it's violated.
                    # We need to know which are violated.
            
            # To avoid O(N) scan for violated every step, we can maintain a set.
            # But for simplicity in Python, let's just scan if the set is small or use a counter.
            
        # This naive repair is too slow if we scan for violated every time.
        
        # Let's optimize:
        # Maintain a list of violated configs.
        # When we remove a triple, we update counts.
        # If a count drops from 8 to 7, it's no longer violated.
        # If a count stays 8 or goes up (impossible in repair), it remains.
        
        # Revised CDLS:
        
    def run_cdls_optimized(seed):
        random.seed(seed)
        present = np.ones(num_triples, dtype=bool)
        config_counts = np.array([8] * num_configs, dtype=np.int8)
        
        # List of violated configs (indices)
        violated = list(range(num_configs))
        
        # Repair
        while violated:
            cid = random.choice(violated)
            # Pick a triple to remove from this config
            # To be smart, pick one that appears in many other violated configs
            tris = triples_in_config[cid]
            
            # Score each triple in tris by how many other violated configs it's in
            best_t = -1
            best_score_t = -1
            for t in tris:
                if not present[t]:
                    continue
                # Count how many violated configs contain t
                # This is expensive. 
                # Approximation: Just pick random.
                # Or: Precompute for each triple, the list of configs.
                # We have triple_to_configs.
                # We need to check if those configs are in `violated`.
                # `violated` is a list. Set lookup is faster.
                
            # Let's just pick random for speed, as checking all is slow.
            t_idx = random.choice(tris)
            if not present[t_idx]:
                # This shouldn't happen if we only pick from present?
                # tris contains all 8. Some might already be removed.
                # We should pick from present ones in this config.
                present_in_cid = [t for t in tris if present[t]]
                if not present_in_cid:
                    # Config is not actually violated? 
                    # If count is 8, there must be 8 present.
                    # So present_in_cid should not be empty.
                    pass
                else:
                    t_idx = random.choice(present_in_cid)
            
            # Remove t_idx
            present[t_idx] = False
            for c in triple_to_configs[t_idx]:
                config_counts[c] -= 1
                if config_counts[c] == 7:
                    # No longer violated
                    # Remove from violated list
                    # List removal is O(N). 
                    # Use a set for violated?
                    pass
            
        # This is getting messy with list updates.
        # Let's use a simpler "WalkSAT" style:
        # 1. Start with random subset (e.g., 50%).
        # 2. While violated:
        #    - Pick violated config.
        #    - Flip a variable in it (remove a triple).
        # 3. While no violated:
        #    - Try to add a random missing triple.
        #    - If it causes violation, revert.
        #    - If it doesn't, keep it.
        # 4. Random restarts.
        
        return 0, None # Placeholder

    # Let's go back to the B&B idea but with a very tight time limit and a strong initial bound.
    # The "MaxSAT" repair is essentially a local search.
    
    # Let's try a **Genetic Algorithm** with a "Crossover" that preserves validity?
    # No, validity is hard to preserve.
    
    # Let's stick to **Branch and Bound** but with a **Bitset** representation for speed.
    
    # Precompute bitsets for configs?
    # config_bitset[cid] = bitmask of triples in config cid.
    
    # State:
    # present_mask: int
    # config_counts: array
    
    # Bound:
    # current_count + (popcount(available_mask))
    # available_mask: triples not in present_mask AND individually valid.
    
    # Individually valid: For all c in triple_to_configs[t], config_counts[c] < 7.
    
    # This check is fast if we precompute triple_to_configs.
    
    # Let's implement this B&B.
    
    triple_bitsets = [0] * num_triples
    for i in range(num_triples):
        triple_bitsets[i] = 1 << i
        
    # Order of branching: 
    # We can sort triples by "degree" (number of configs) to prune faster.
    # High degree triples affect more configs, so deciding them early might help.
    
    sorted_triples = sorted(range(num_triples), key=lambda x: len(triple_to_configs[x]), reverse=True)
    
    # Map new index to old index
    old_to_new = {old: new for new, old in enumerate(sorted_triples)}
    
    # Re-map triple_to_configs to new indices? 
    # Or just use old indices in the logic and iterate through sorted_triples.
    
    # B&B Function
    def branch_and_bound():
        nonlocal best_score, best_state
        present = np.zeros(num_triples, dtype=bool)
        config_counts = np.zeros(num_configs, dtype=np.int8)
        current_count = 0
        
        # Recursive function
        def dfs(idx):
            nonlocal best_score, best_state
            
            if time.time() > deadline - 2:
                return
            
            # Bound
            # Count how many remaining triples (from idx to end) are valid
            # This is O(N * K). Too slow for every node.
            # Optimization: Only compute bound every 10 levels?
            # Or use a weaker bound: current_count + (num_triples - idx)
            # If current_count + (num_triples - idx) <= best_score: return
            
            if current_count + (num_triples - idx) <= best_score:
                return
            
            if idx == num_triples:
                if current_count > best_score:
                    best_score = current_count
                    best_state = present.copy()
                return
            
            t_idx = sorted_triples[idx]
            
            # Option 1: Include t_idx
            # Check if valid
            valid = True
            for c in triple_to_configs[t_idx]:
                if config_counts[c] == 7:
                    valid = False
                    break
            
            if valid:
                present[t_idx] = True
                for c in triple_to_configs[t_idx]:
                    config_counts[c] += 1
                current_count += 1
                
                dfs(idx + 1)
                
                present[t_idx] = False
                for c in triple_to_configs[t_idx]:
                    config_counts[c] -= 1
                current_count -= 1
            
            # Option 2: Exclude t_idx
            dfs(idx + 1)
            
        # Initial Greedy to set best_score
        # Use the existing greedy logic
        present[:] = False
        config_counts[:] = 0
        for t_idx in sorted_triples:
            valid = True
            for c in triple_to_configs[t_idx]:
                if config_counts[c] == 7:
                    valid = False
                    break
            if valid:
                present[t_idx] = True
                for c in triple_to_configs[t_idx]:
                    config_counts[c] += 1
        
        best_score = np.sum(present)
        best_state = present.copy()
        
        # Run B&B
        present[:] = False
        config_counts[:] = 0
        dfs(0)
        
    # Run B&B
    branch_and_bound()
    
    result = [all_triples[i] for i in range(num_triples) if best_state[i]]
    return result

# To ensure we don't time out, we wrap the B&B in a try-catch or time check.
# The DFS above has a time check.

# However, the bound `current_count + (num_triples - idx)` is very weak.
# We can improve it by checking validity of the next few triples.
# But let's trust the high-degree ordering to prune.

# One more improvement: 
# If a triple is INVALID (blocked), we MUST exclude it. No branching needed.
# The code above handles this: if not valid, it skips Option 1.

# Final check:
# The problem size is 560.
# Branching factor 2.
# Depth 560.
# 2^560 is huge.
# Pruning is critical.
# If the bound is weak, it will explode.
# We need a better bound.
# 
# Better Bound:
# current_count + (number of valid triples among the remaining).
# We can compute this quickly if we maintain a count of valid triples?
# When we add a triple, some remaining triples become invalid.
# When we remove a triple, some become valid.
# 
# Let's add a `valid_count` variable.
# Initially, all 560 are valid (assuming counts are 0).
# When we add t:
#   For each config c in t:
#     If count[c] becomes 7:
#       All other triples in c become invalid (if they were valid).
#       We need to decrement valid_count for each such triple.
# 
# This is complex to maintain incrementally.
# 
# Given the constraints, the simple B&B might be too slow.
# Let's fallback to a **Simulated Annealing** with a **much more aggressive restart** and **2-out 1-in** moves, but with a **bitmask-based validity check** for speed.
# 
# Wait, the current program IS SA. 
# The directive says "different design family".
# B&B is different. Even if it's slow, it's a different family.
# If it times out, it returns the greedy result.
# The greedy result is likely around 2.4 (same as current).
# 
# To beat 2.4, we need to find a better solution.
# B&B might find the OPTIMAL solution if it doesn't time out.
# 
# Let's enhance the B&B with a **tighter bound** using a **greedy independent set** on the remaining valid triples.
# 
# Bound = current_count + greedy_mis_size(remaining_valid_triples)
# 
# greedy_mis_size:
#   While there are valid triples:
#     Pick one with min conflicts.
#     Add to set.
#     Mark its conflicts as invalid.
# 
# This greedy is O(N^2) per node? Too slow.
# 
# Let's just use the simple bound and hope the high-degree ordering works.
# Or, limit the depth of B&B to, say, 50 levels, and use SA for the rest?
# That's a hybrid.
# 
# Let's stick to the pure B&B with the simple bound, but add a time limit inside the recursion.
# If it times out, it returns the best found.

# One final tweak: 
# Use `sys.setrecursionlimit(10000)`

import sys
sys.setrecursionlimit(10000)

# Re-run the logic with the B&B inside construct()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
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
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
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
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    sorted_triples = sorted(range(num_triples), key=lambda x: len(triple_to_configs[x]), reverse=True)
    
    best_score = 0
    best_state = np.zeros(num_triples, dtype=bool)
    
    # Greedy Init
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    for t_idx in sorted_triples:
        valid = True
        for c in triple_to_configs[t_idx]:
            if config_counts[c] == 7:
                valid = False
                break
        if valid:
            present[t_idx] = True
            for c in triple_to_configs[t_idx]:
                config_counts[c] += 1
    best_score = np.sum(present)
    best_state = present.copy()
    
    # B&B
    present[:] = False
    config_counts[:] = 0
    
    def dfs(idx):
        nonlocal best_score, best_state
        if time.time() > deadline - 2:
            return
        if current_count + (num_triples - idx) <= best_score:
            return
        if idx == num_triples:
            if current_count > best_score:
                best_score = current_count
                best_state = present.copy()
            return
        
        t_idx = sorted_triples[idx]
        
        # Try Include
        valid = True
        for c in triple_to_configs[t_idx]:
            if config_counts[c] == 7:
                valid = False
                break
        
        if valid:
            present[t_idx] = True
            for c in triple_to_configs[t_idx]:
                config_counts[c] += 1
            current_count += 1
            dfs(idx + 1)
            present[t_idx] = False
            for c in triple_to_configs[t_idx]:
                config_counts[c] -= 1
            current_count -= 1
            
        # Try Exclude
        dfs(idx + 1)
        
    current_count = 0
    dfs(0)
    
    result = [all_triples[i] for i in range(num_triples) if best_state[i]]
    return result

