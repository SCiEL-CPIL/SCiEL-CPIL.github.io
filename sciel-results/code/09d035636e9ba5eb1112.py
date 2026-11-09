"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # 1. Precompute prime factors and compatibility
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    # Build conflict graph using bitsets
    # conflict[i] is a bitset of all j that conflict with i (including i itself)
    conflict = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if i == j:
                conflict[i] |= (1 << i)
            elif prime_factors[i] & prime_factors[j]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # weights[i] = i (the value of including integer i)
    weights = list(range(N + 1))
    
    # 2. Initial solution: greedy by descending weight
    initial_set = []
    current_mask = 0
    for num in range(N, 0, -1):
        if (current_mask & conflict[num]) == 0:
            initial_set.append(num)
            current_mask |= (1 << num)
    
    best_sum = sum(initial_set)
    best_set = list(initial_set)
    
    # 3. Try to use NetworkX for exact Maximum Weight Independent Set
    # via Maximum Weight Clique on the complement graph
    try:
        import networkx as nx
        
        # Build the complement graph (edges where numbers are coprime)
        # The complement of the conflict graph: edge (i,j) exists if gcd(i,j)==1
        comp_graph = nx.Graph()
        comp_graph.add_nodes_from(range(1, N + 1))
        
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                # i and j are coprime if they don't share a prime factor
                if not (prime_factors[i] & prime_factors[j]):
                    comp_graph.add_edge(i, j)
        
        # Set weights
        for node in comp_graph.nodes:
            comp_graph.nodes[node]['weight'] = node
        
        # Find maximum weight clique in the complement graph
        # This is equivalent to maximum weight independent set in the original conflict graph
        if time.time() < deadline - 5:
            mw_clique = nx.algorithms.approximation.max_weight_clique(comp_graph)
            clique_sum = sum(comp_graph.nodes[n]['weight'] for n in mw_clique)
            
            if clique_sum > best_sum:
                best_sum = clique_sum
                best_set = sorted([int(n) for n in mw_clique])
        
        # If we have time, also try the exact version
        if time.time() < deadline - 10:
            try:
                exact_clique = nx.algorithms.approximation.max_weight_clique(
                    comp_graph, 
                    seed=42
                )
                exact_sum = sum(comp_graph.nodes[n]['weight'] for n in exact_clique)
                if exact_sum > best_sum:
                    best_sum = exact_sum
                    best_set = sorted([int(n) for n in exact_clique])
            except Exception:
                pass
                
    except ImportError:
        pass
    except Exception:
        pass
    
    # 4. Local search to polish the result
    def try_improve(current_mask, current_sum):
        """Run local search on current state, return improved (mask, sum)."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try 1-for-0 insertions
            for num in range(N, 0, -1):
                if current_mask & (1 << num):
                    continue
                if (current_mask & conflict[num]) == 0:
                    current_mask |= (1 << num)
                    current_sum += num
                    improved = True
            
            # Try 1-for-1 swaps (remove one, add one larger)
            if not improved:
                for num_out in range(1, N + 1):
                    if not (current_mask & (1 << num_out)):
                        continue
                    mask_without_out = current_mask & ~(1 << num_out)
                    for num_in in range(N, num_out, -1):
                        if current_mask & (1 << num_in):
                            continue
                        if (mask_without_out & conflict[num_in]) == 0:
                            current_mask = mask_without_out | (1 << num_in)
                            current_sum = current_sum - num_out + num_in
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 2-for-1 swaps (remove two small, add one large)
            if not improved:
                for num_out1 in range(1, N + 1):
                    if not (current_mask & (1 << num_out1)):
                        continue
                    for num_out2 in range(num_out1 + 1, N + 1):
                        if not (current_mask & (1 << num_out2)):
                            continue
                        mask_without = current_mask & ~(1 << num_out1) & ~(1 << num_out2)
                        for num_in in range(N, max(num_out1, num_out2), -1):
                            if current_mask & (1 << num_in):
                                continue
                            if (mask_without & conflict[num_in]) == 0:
                                new_sum = current_sum - num_out1 - num_out2 + num_in
                                if new_sum > current_sum:
                                    current_mask = mask_without | (1 << num_in)
                                    current_sum = new_sum
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try 1-for-2 swaps (remove one, add two)
            if not improved:
                for num_out in range(1, N + 1):
                    if not (current_mask & (1 << num_out)):
                        continue
                    mask_without_out = current_mask & ~(1 << num_out)
                    added = False
                    for num_in1 in range(N, 0, -1):
                        if current_mask & (1 << num_in1):
                            continue
                        if (mask_without_out & conflict[num_in1]) != 0:
                            continue
                        mask_with_in1 = mask_without_out | (1 << num_in1)
                        for num_in2 in range(num_in1 - 1, 0, -1):
                            if current_mask & (1 << num_in2):
                                continue
                            if (mask_with_in1 & conflict[num_in2]) != 0:
                                continue
                            new_sum = current_sum - num_out + num_in1 + num_in2
                            if new_sum > current_sum:
                                current_mask = mask_with_in1 | (1 << num_in2)
                                current_sum = new_sum
                                improved = True
                                added = True
                                break
                        if added:
                            break
                    if added:
                        break
        
        return current_mask, current_sum
    
    # Run local search on the best solution found
    final_mask = 0
    for num in best_set:
        final_mask |= (1 << num)
    final_sum = sum(best_set)
    
    final_mask, final_sum = try_improve(final_mask, final_sum)
    if final_sum > best_sum:
        best_sum = final_sum
        best_set = [i for i in range(N + 1) if final_mask & (1 << i)]
    
    # Convert best_set to list
    final_set = [int(x) for x in best_set]
    
    return final_set

# EVOLVE-BLOCK-END
