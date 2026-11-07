# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    
    # Deadline guard: 85s safety margin for 90s limit
    start_time = time.time()
    max_time = 85.0
    
    # Pre-compute all 2^12 subsets
    all_subsets = [i for i in range(1 << n)]
    S = len(all_subsets) # 4096
    
    # Pre-compute subset relationships
    # is_subset[a][b] is 1 if all_subsets[a] is a subset of all_subsets[b]
    # Note: In ILP, we maximize sum_{i<j} (x_i x_j (is_subset[i][j] + is_subset[j][i] - 1_{i==j}))
    # Since we require distinct sets, i != j is enforced by x_i + x_j <= 1 if i==j? 
    # No, x_i are 0/1. Distinctness: we cannot pick the same subset twice.
    # The variables are y_{k, s} = 1 if the k-th set in our family is the subset s.
    # This is an assignment problem structure.
    
    # Variables:
    # y_{k, s} for k in 0..m-1, s in 0..S-1.
    # Total vars: m * S = 40 * 4096 = 163,840.
    # This is large for scipy.milp.
    
    # Alternative formulation:
    # Let x_s be 1 if subset s is in the family.
    # We need exactly m subsets. Sum(x_s) = m.
    # Objective: Maximize sum_{s < t} (x_s x_t (is_subset[s][t] + is_subset[t][s]))
    # This is a Quadratic Integer Program (QIP).
    # scipy.optimize.milp handles Linear constraints and Linear objective.
    # It does NOT handle quadratic objectives directly.
    
    # However, we can linearize the product x_s x_t using auxiliary variables z_{s,t}.
    # z_{s,t} = 1 iff x_s = 1 and x_t = 1.
    # Constraints:
    # z_{s,t} <= x_s
    # z_{s,t} <= x_t
    # z_{s,t} >= x_s + x_t - 1
    # z_{s,t} is binary (or continuous [0,1] since objective is max and coeffs positive? 
    # Actually, if we maximize sum c_{s,t} z_{s,t} with c >= 0, and z <= min(x_s, x_t), 
    # the optimal solution will push z up to min(x_s, x_t) if c > 0.
    # So z can be continuous [0,1].
    
    # Number of pairs (s,t) with s < t: ~ 8 million.
    # Variables: S (x) + ~8M (z).
    # Constraints: 3 * 8M + 1 (sum x) + bounds.
    # This is WAY too large for scipy.milp in 60s.
    
    # Revised Strategy:
    # The structure of the problem suggests that optimal solutions are "dense" in the lattice.
    # For n=12, the maximum number of comparable pairs for a family of size m
    # is achieved by taking an initial segment of the Boolean lattice ordered by size (or inclusion).
    # Specifically, LYM inequality and related results suggest that "initial segments" of the
    # inclusion order (or size order) are optimal for maximizing comparable pairs.
    # 
    # Let's test the hypothesis that the optimal set is the m smallest subsets (by size, then lex).
    # Or, more generally, we can use a greedy construction that picks the subset with the highest
    # "comparability potential" with the already chosen set, but this is what SA does.
    
    # Let's try a different structural insight:
    # The problem is equivalent to finding a subgraph of the comparability graph of the Boolean lattice B_n
    # that is a clique? No, a set of vertices with max edges.
    # This is the "Maximum Induced Subgraph" problem on the comparability graph, which is NP-hard.
    # However, for B_n, the comparability graph is a perfect graph? No, comparability graphs are perfect.
    # Wait, the comparability graph of a poset is a perfect graph.
    # If the graph is perfect, the Maximum Weight Clique can be solved in polynomial time?
    # No, Max Clique is NP-hard even for perfect graphs? 
    # Actually, Max Clique in Perfect Graphs is solvable in polynomial time (Grötschel, Lovasz, Schrijver).
    # BUT, we are not looking for a clique. We are looking for a subset of vertices that maximizes the NUMBER OF EDGES (comparable pairs).
    # This is the "Maximum Induced Subgraph with k vertices" problem, or "Densest k-subgraph".
    # Densest k-subgraph is NP-hard even for special graphs.
    
    # So ILP is the right path, but the size is the issue.
    
    # Let's try a smaller, smarter ILP.
    # Symmetry breaking: We can assume the sets are ordered by size?
    # If A is a subset of B, size(A) <= size(B).
    # In an optimal solution, can we assume the sets are sorted by size?
    # Yes, the family is just a set of subsets.
    # But the "comparable" relation depends on content, not just size.
    
    # Let's use the property that for any two sets A, B, if they are incomparable, we lose an edge.
    # To maximize edges, we want to avoid incomparable pairs.
    # This suggests choosing sets from a chain.
    # A chain in B_12 has length 13 (sizes 0 to 12).
    # We need 40 sets.
    # We can't have a single chain of 40.
    # We need to branch.
    
    # Heuristic: The optimal family for maximizing comparable pairs is likely composed of
    # sets that are "close" in the lattice.
    
    # Let's try a Multi-Start Local Search but with a much better initialization:
    # Initialize with the 40 sets that have the highest "centrality" in the comparability graph?
    # Or, initialize with sets of sizes around m/2? No, small sets have many supersets.
    # Set of size 0 is comparable to all 4095 sets.
    # Set of size 1 is comparable to all sets containing it (2^11) + sets not containing it (2^11 - 1? no).
    # Number of comparable sets for S: 2^{|S|} (subsets) + 2^{n-|S|} (supersets) - 1 (S itself).
    # For n=12:
    # |S|=0: 1 + 4096 - 1 = 4096
    # |S|=1: 2 + 2048 - 1 = 2049
    # |S|=2: 4 + 1024 - 1 = 1027
    # |S|=6: 64 + 64 - 1 = 127
    # So small sets are "hubs".
    # If we pick the 40 smallest sets (by size), they are all comparable to each other?
    # No. {0} and {1} are incomparable.
    # But {0} is comparable to {0,1}.
    
    # Let's try a Greedy Construction:
    # 1. Start with empty family.
    # 2. At each step, add the set S (not already in family) that maximizes the number of
    #    comparable pairs with the existing family.
    # 3. If ties, prefer smaller size (more potential for future growth? or just deterministic).
    
    # This is a greedy algorithm. It might get stuck in local optima, but it's a different
    # structural approach from SA (which is random walk).
    # To improve, we can run this greedy algorithm multiple times with different starting seeds
    # or different tie-breaking rules, and then refine with a fast local search (not full SA, but 
    # hill climbing with restarts).
    
    # However, the prompt asks for a STRUCTURAL change.
    # The current program is SA.
    # Let's try **Greedy + Local Search** with a specific initialization based on the "hub" property.
    
    # Actually, there is a known result for this problem:
    # The maximum number of comparable pairs in a family of m subsets of an n-set
    # is achieved by the family consisting of the m smallest subsets (by size).
    # Let's verify this hypothesis.
    # If true, the solution is deterministic and trivial!
    # Let's check for small n.
    # n=2, m=3.
    # Subsets: {}, {0}, {1}, {0,1}
    # Sizes: 0, 1, 1, 2
    # Smallest 3: {}, {0}, {1}.
    # Pairs: ({}, {0}) comp, ({}, {1}) comp, ({0}, {1}) NOT comp. Total 2.
    # Alternative: {}, {0}, {0,1}.
    # Pairs: ({}, {0}) comp, ({}, {0,1}) comp, ({0}, {0,1}) comp. Total 3.
    # So "smallest by size" is NOT optimal.
    # The chain {}, {0}, {0,1} is better.
    
    # So we want to form chains.
    # A chain of length L has L(L-1)/2 comparable pairs.
    # For m=40, if we could form one chain, we'd have 40*39/2 = 780 pairs.
    # But max chain length is 13.
    # So we need multiple chains.
    # To maximize total pairs, we want the chains to be as long as possible, and we want
    # comparability BETWEEN chains?
    # If A is in Chain 1 and B is in Chain 2, and A < B, that's an extra pair.
    # So we want the chains to be "nested" or "interleaved" such that elements of lower chains
    # are subsets of elements in higher chains.
    
    # This suggests a construction based on **Initial Segments of the Poset**.
    # Order the subsets by a linear extension of the poset (e.g., by size, then lex).
    # Take the first m elements.
    # Let's test this.
    # Order: 0, 1, 2, ..., 4095 (binary).
    # This is not a linear extension of inclusion.
    # A linear extension of inclusion: Sort by size, then lex.
    # Let's try this deterministic construction first. It's O(1) if we precompute.
    
    # Precompute all subsets, sort by (size, lex).
    # Take first 40.
    # Count pairs.
    
    # If this score is high, we are done.
    # If not, we use this as a seed for a fast local search.
    
    # Let's implement the deterministic "Initial Segment" construction.
    
    subsets = []
    for i in range(1 << n):
        # Count bits
        s = bin(i).count('1')
        subsets.append((s, i))
        
    # Sort by size, then by value (lex)
    subsets.sort()
    
    # Take first m
    selected_masks = [x[1] for x in subsets[:m]]
    
    # Count score
    def count_pairs(masks):
        c = 0
        m = len(masks)
        for i in range(m):
            for j in range(i+1, m):
                if (masks[i] & masks[j]) == masks[i] or (masks[i] & masks[j]) == masks[j]:
                    c += 1
        return c
    
    base_score = count_pairs(selected_masks)
    
    # Now, let's try a few other deterministic orderings.
    # 1. Sort by size, then by "popcount of complement"? No.
    # 2. Sort by size, then by max element?
    
    # Let's try a Local Search starting from this base.
    # But the directive says "STRUCTURAL change".
    # The current program is SA.
    # I will use **Greedy Construction** to build the initial solution, followed by **Hill Climbing**.
    # The Greedy step is fundamentally different from Random/Chain init.
    
    # Greedy Construction:
    # Start with empty set.
    # For k in 1..m:
    #   Find subset S not in current family that maximizes comparability with current family.
    #   Add S.
    
    # This is O(m * 2^n * m) = 40 * 4096 * 40 ~ 6.5M operations. Fast.
    
    family = []
    # Precompute comparability matrix? 4096x4096 is 16M booleans. 16MB. OK.
    # But we only need to check against current family.
    
    # To speed up greedy, we can precompute for each subset S, the list of subsets it is comparable to.
    # comp_mask[S] = bitmask of subsets comparable to S.
    # Then score(S, family) = popcount(comp_mask[S] & family_mask).
    
    # Let's build this.
    comp_masks = [0] * (1 << n)
    for i in range(1 << n):
        for j in range(1 << n):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                comp_masks[i] |= (1 << j)
    
    # This precomputation is 16M ops. ~0.1s.
    
    current_family = []
    family_mask = 0
    
    for _ in range(m):
        best_s = -1
        best_score = -1
        # Iterate all subsets
        for s in range(1 << n):
            if family_mask & (1 << s):
                continue
            # Count how many in current_family are comparable to s
            # This is popcount(comp_masks[s] & family_mask)
            # Python int bit_count is fast.
            sc = (comp_masks[s] & family_mask).bit_count()
            # Tie-breaking: prefer smaller size? Or smaller index?
            # Let's prefer smaller size to keep the family "dense" at the bottom.
            size = bin(s).count('1')
            # We want to maximize sc. If tie, minimize size.
            if sc > best_score or (sc == best_score and (best_s == -1 or size < bin(best_s).count('1'))):
                best_score = sc
                best_s = s
        
        if best_s == -1:
            break
            
        current_family.append(best_s)
        family_mask |= (1 << best_s)
        
    greedy_score = count_pairs(current_family)
    
    # Now we have a greedy solution.
    # Let's try to improve it with Hill Climbing.
    # Move: Replace one set S in family with a new set S'.
    # Delta = comparability(S', family \ {S}) - comparability(S, family \ {S})
    
    # Precompute for current family:
    # For each i in family, how many others are comparable to it?
    # deg[i]
    
    def get_degrees(fam):
        degs = [0] * len(fam)
        for i in range(len(fam)):
            degs[i] = (comp_masks[fam[i]] & family_mask).bit_count() - 1 # Exclude self
        return degs
    
    # Wait, family_mask changes.
    
    # Let's run a few restarts of Hill Climbing from the Greedy solution.
    # And also from the "Initial Segment" solution.
    
    candidates = [
        selected_masks, # Initial Segment
        current_family  # Greedy
    ]
    
    best_overall = max(candidates, key=count_pairs)
    best_score = count_pairs(best_overall)
    
    # Hill Climbing
    # We have time. Let's run HC for the remaining time.
    # HC is fast.
    
    def hill_climb(start_fam, time_limit):
        fam = list(start_fam)
        fam_set = set(fam)
        fam_mask = 0
        for s in fam:
            fam_mask |= (1 << s)
            
        current_score = count_pairs(fam)
        
        # Precompute degrees
        # deg[i] = number of j != i such that fam[i] and fam[j] are comparable
        degs = [0] * m
        for i in range(m):
            degs[i] = (comp_masks[fam[i]] & fam_mask).bit_count() - 1
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(m):
                old_s = fam[i]
                # Try to replace old_s with a new_s
                # We want new_s that maximizes comparability with (fam \ {old_s})
                # Score(new_s) = (comp_masks[new_s] & (fam_mask ^ (1 << old_s))).bit_count()
                # Current contribution of old_s is degs[i].
                # Delta = Score(new_s) - degs[i]
                
                # To find best new_s, we can iterate all 4096 subsets.
                best_new_s = -1
                best_delta = -1
                temp_mask = fam_mask ^ (1 << old_s)
                
                for s in range(1 << n):
                    if s in fam_set:
                        continue
                    sc = (comp_masks[s] & temp_mask).bit_count()
                    delta = sc - degs[i]
                    if delta > best_delta:
                        best_delta = delta
                        best_new_s = s
                
                if best_new_s != -1 and best_delta > 0:
                    # Accept
                    fam_set.remove(old_s)
                    fam_set.add(best_new_s)
                    fam[i] = best_new_s
                    fam_mask = (fam_mask ^ (1 << old_s)) | (1 << best_new_s)
                    
                    # Update degrees
                    # degs[i] becomes best_delta + degs[i]? No.
                    # New degs[i] = sc (comparability with others)
                    # Other degrees j:
                    # If old_s and fam[j] were comparable, degs[j] decreases by 1.
                    # If best_new_s and fam[j] are comparable, degs[j] increases by 1.
                    
                    for j in range(m):
                        if j == i:
                            continue
                        # Check old_s vs fam[j]
                        if (old_s & fam[j]) == old_s or (old_s & fam[j]) == fam[j]:
                            degs[j] -= 1
                        # Check best_new_s vs fam[j]
                        if (best_new_s & fam[j]) == best_new_s or (best_new_s & fam[j]) == best_new_s:
                            degs[j] += 1
                            
                    degs[i] = sc
                    current_score += best_delta
                    improved = True
                    # If we made a change, we should restart the i loop or continue?
                    # Continue is fine, but degrees changed.
                    # Actually, if we change i, degrees of others change.
                    # So we should probably break and restart the i loop to be safe,
                    # or just continue and rely on the fact that we check all i.
                    # But degs are updated, so subsequent i checks are valid.
                    
        return fam, current_score
    
    # Run Hill Climbing on candidates
    # Time budget: 80s
    hc_time_limit = time.time() + 78.0
    
    # Run on Greedy
    hc_fam, hc_score = hill_climb(current_family, hc_time_limit)
    if hc_score > best_score:
        best_score = hc_score
        best_overall = hc_fam
        
    # Run on Initial Segment
    if time.time() < hc_time_limit:
        hc_fam2, hc_score2 = hill_climb(selected_masks, hc_time_limit)
        if hc_score2 > best_score:
            best_score = hc_score2
            best_overall = hc_fam2
            
    # If time remains, try random restarts of Hill Climbing?
    # Or just return best.
    
    # Convert to lists of ints
    result = []
    for mask in best_overall:
        s = []
        for b in range(n):
            if mask & (1 << b):
                s.append(b)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
