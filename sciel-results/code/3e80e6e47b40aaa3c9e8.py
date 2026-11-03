"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Generate Weight 4 vectors only
    # Weight 4 vectors have norm^2 = 4.
    # Distance condition: dist^2 >= max(norm^2) = 4.
    # dist^2 = 4 + 4 - 2*dot = 8 - 2*dot >= 4  =>  2*dot <= 4  =>  dot <= 2.
    # For weight 4 vectors, dot product is integer.
    # If dot > 2, they conflict.
    # Possible dot products for weight 4 vectors:
    # -4 (opposite), -2, 0, 2, 4 (same)
    # Conflict if dot > 2. So dot = 4 is conflict.
    # Wait, if dot=2, dist^2 = 8-4=4, which is >= 4. OK.
    # If dot=3? Not possible for weight 4 vectors with +/-1 entries?
    # Let's check: v=(1,1,1,1,0..), w=(1,1,1,-1,0..). dot = 1+1+1-1 = 2.
    # v=(1,1,1,1,0..), w=(1,1,-1,-1,0..). dot = 1+1-1-1 = 0.
    # v=(1,1,1,1,0..), w=(1,1,1,1,0..). dot = 4.
    # v=(1,1,1,1,0..), w=(1,1,1,0,1,0..). dot = 1+1+1+0+0 = 3.
    # Ah, dot=3 is possible if they share 3 coordinates with same sign and 1 with different sign?
    # v: pos 0,1,2,3. w: pos 0,1,2,4.
    # v[0]=1, v[1]=1, v[2]=1, v[3]=1.
    # w[0]=1, w[1]=1, w[2]=1, w[4]=1.
    # dot = 1+1+1+0 = 3.
    # dist^2 = 8 - 2(3) = 2 < 4. Conflict!
    # So conflict if dot > 2.
    
    candidates = []
    norms_sq = []
    # Map from vector tuple to index
    vec_to_idx = {}

    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            vec_to_idx[tuple(v)] = len(candidates) - 1

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)

    # Compute conflicts
    # We can exploit structure: two weight 4 vectors conflict if their dot product > 2.
    # dot = sum(v[i]*w[i]).
    # Instead of full matrix, we can precompute neighbors more efficiently or just use the matrix since N is ~ 3300.
    # 3300^2 = ~10M ops. Very fast.
    dots = np.dot(candidates, candidates.T)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    # Condition: dist_sq >= max(norms_sq[i], norms_sq[j])
    # Since all norms_sq are 4, condition is dist_sq >= 4.
    # Conflict if dist_sq < 4.
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)

    # Precompute neighbors
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())

    # GRASP Implementation
    # 1. Construction: Restricted Candidate List (RCL)
    # 2. Local Search: 1-for-1 swaps (and maybe 2-for-1 if time permits, but directive says 1-for-1)
    
    best_selected_set = set()
    best_score = 0

    # Helper to evaluate a set
    def is_valid_set(s):
        # Already guaranteed by construction/local search, but just in case
        for i in s:
            if neighbors[i] & s:
                return False
        return True

    def local_search(s):
        """
        1-for-1 swap local search.
        Try to swap a node in s with a node not in s to increase size or maintain size with better properties?
        Actually, standard independent set local search tries to add nodes.
        If full, try swapping.
        However, simple "add if possible" is a 1-for-0 move.
        The directive specifies "1-for-1 swap local search to escape local optima".
        This implies we are stuck at a local maximum where no node can be added.
        We try to remove u and add v. This keeps size constant. How does this help?
        It helps if v allows adding another node w later? Or maybe the "1-for-1" is actually "remove u, add v AND w"?
        Usually, to increase size, you need 1-for-2 or 2-for-3 swaps.
        If we strictly do 1-for-1, we stay at same size.
        However, often "local search" in MIS context implies:
        1. Try to add any node.
        2. If no node can be added, try to swap (remove 1, add 1) to change the structure,
           potentially allowing an addition in the next step?
           Or maybe the swap is just to find a *different* local maximum of the same size?
           That doesn't increase score.
           
           Let's re-read: "refines them using a 1-for-1 swap local search to escape local optima".
           If we escape a local optimum of size K, we might land in a basin that leads to K+1?
           Or maybe the "1-for-1" is a typo in my interpretation and it means "swap one out for one in" which is neutral,
           but combined with "add if possible" loops, it explores the space of maximal independent sets.
           
           Let's implement:
           1. Greedy fill (add if possible).
           2. Loop:
              a. Try to add any node. If added, restart loop.
              b. If no add possible, try 1-for-1 swaps.
                 - Pick u in S.
                 - Pick v not in S, not neighbor of S\{u}.
                 - If v is not neighbor of u (trivially true if v not in S and S is independent? No, v could be neighbor of u).
                 - Actually, for v to be added to S\{u}, v must not be neighbor of any node in S\{u}.
                 - If such v exists, do the swap. This changes S.
                 - Then try to add again.
           
           This is a valid heuristic.
        """
        s = set(s)
        # First, fill greedily
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Find all nodes that can be added
            # Optimization: maintain list of candidates
            for v in range(n_candidates):
                if v in s:
                    continue
                if not (neighbors[v] & s):
                    s.add(v)
                    changed = True
                    # Restart scan? Or continue?
                    # Continuing is fine, we just added one.
                    # But adding one might block others.
                    # Let's just do a full pass.
            
            # If no change, try swaps
            if not changed:
                # Try 1-for-1 swaps
                swapped = False
                s_list = list(s)
                random.shuffle(s_list)
                for u in s_list:
                    if time.time() > deadline - 2.0:
                        break
                    # Try to find v such that v can replace u
                    # v must be:
                    # 1. Not in s
                    # 2. Not neighbor of any node in s \ {u}
                    # 3. (Optional) Not neighbor of u? If v is neighbor of u, it's fine, we remove u.
                    
                    # Pre-compute conflict set of s \ {u}
                    # This is expensive if done naively.
                    # Let's iterate through potential v's?
                    # Or iterate through v's that are neighbors of u?
                    # If v is not neighbor of u, and not neighbor of s\{u}, then v is isolated from s.
                    # Then v could have been added in the first place!
                    # So if v is not neighbor of s\{u} AND v is not neighbor of u, then v is not neighbor of s.
                    # Thus v could be added directly.
                    # Since we are in the "no add possible" state, there is NO v not neighbor of s.
                    # Therefore, any valid replacement v MUST be a neighbor of u.
                    
                    # So we only need to check v in neighbors[u]
                    # Check if v is not neighbor of s \ {u}
                    
                    for v in neighbors[u]:
                        if v in s:
                            continue
                        # Check if v conflicts with s \ {u}
                        conflict = False
                        for w in s:
                            if w == u:
                                continue
                            if v in neighbors[w]:
                                conflict = True
                                break
                        if not conflict:
                            # Swap u with v
                            s.remove(u)
                            s.add(v)
                            swapped = True
                            break
                    if swapped:
                        break
                if not swapped:
                    break # No moves possible
        
        return s

    # GRASP Loop
    random.seed(42)
    np.random.seed(42)
    
    # Parameters
    alpha = 0.3 # RCL parameter
    num_iterations = 0
    
    while time.time() < deadline - 10.0:
        # 1. Construction Phase
        available = set(range(n_candidates))
        selected = set()
        
        while available:
            # Calculate degrees in available subgraph
            # deg(v) = |neighbors[v] & available|
            # To avoid O(N^2) every step, we can update degrees incrementally?
            # N is ~3300. N^2 is 10M.
            # If we do this for every node added (up to ~100 nodes), that's 100 * 10M = 1B ops. Too slow.
            
            # Optimization:
            # We only need to consider nodes in `available`.
            # We can maintain a list of candidates.
            
            # Simple approach:
            # If available is large, sample? No, RCL needs exact degrees.
            # Let's optimize the degree calculation.
            
            # Actually, 3300 nodes.
            # In each step, we calculate degrees for all `available` nodes.
            # Sum of sizes of `available` over all steps.
            # If we add 1 node, we remove it and its neighbors from `available`.
            # Average degree is ~50?
            # So we remove ~50 nodes per step.
            # Number of steps ~ 3300/50 = 66.
            # Cost per step: O(|available| * avg_deg)? No, just O(|available|) if we have precomputed neighbor sets.
            # len(neighbors[v] & available) is O(min(deg(v), |available|)).
            # Total cost ~ Sum_{steps} |available|_step * avg_cost.
            # This should be manageable.
            
            if not available:
                break
                
            # Compute degrees
            degrees = {}
            for v in available:
                degrees[v] = len(neighbors[v] & available)
            
            max_deg = max(degrees.values())
            min_deg = min(degrees.values())
            
            # RCL threshold
            if max_deg == min_deg:
                rcl = list(available)
            else:
                threshold = min_deg + alpha * (max_deg - min_deg)
                rcl = [v for v, d in degrees.items() if d <= threshold]
            
            # Select randomly from RCL
            if not rcl:
                v = random.choice(list(available))
            else:
                v = random.choice(rcl)
            
            selected.add(v)
            available.discard(v)
            # Remove neighbors of v from available
            for n in neighbors[v]:
                if n in available:
                    available.discard(n)
        
        # 2. Local Search Phase
        refined = local_search(selected)
        
        if len(refined) > best_score:
            best_score = len(refined)
            best_selected_set = set(refined)
            
        num_iterations += 1
        # Optional: stop if no improvement for long time? 
        # Or just run until deadline.
        
    # Final check: try to add more to best_selected_set
    if best_selected_set:
        # Greedy fill
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for v in range(n_candidates):
                if v in best_selected_set:
                    continue
                if not (neighbors[v] & best_selected_set):
                    best_selected_set.add(v)
                    changed = True

    # Convert to final array
    if not best_selected_set:
        # Fallback
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)

    return candidates[list(best_selected_set)]
# EVOLVE-BLOCK-END
