import random
import time
import math

def construct():
    d = 7
    n_vertices = 2 ** d
    total_edges = d * (2 ** (d - 1))

    # Generate all edges with canonical representation
    edges = []
    edge_index = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                edge_index[(x, y)] = len(edges) - 1
                edge_index[(y, x)] = len(edges) - 1

    # Precompute for each edge: the list of 3-tuples of other edge indices
    # such that if all three are selected, this edge cannot be added (would complete a C4)
    conflicts = [[] for _ in range(total_edges)]

    for idx, (u, v) in enumerate(edges):
        diff = u ^ v
        i_bit = diff.bit_length() - 1

        for j in range(d):
            if j == i_bit:
                continue
            bit_j = 1 << j
            u_prime = u ^ bit_j
            v_prime = v ^ bit_j

            e1 = (min(u, u_prime), max(u, u_prime))
            e2 = (min(v, v_prime), max(v, v_prime))
            e3 = (min(u_prime, v_prime), max(u_prime, v_prime))

            idx1 = edge_index[e1]
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            conflicts[idx].append((idx1, idx2, idx3))

    # For each edge, also precompute the set of edges that participate in any conflict with it
    # This helps for local search
    conflict_partners = [set() for _ in range(total_edges)]
    for idx in range(total_edges):
        for (i1, i2, i3) in conflicts[idx]:
            conflict_partners[idx].add(i1)
            conflict_partners[idx].add(i2)
            conflict_partners[idx].add(i3)

    start_time = time.time()
    time_limit = 115
    rng = random.Random(12345)

    best_count = 0
    best_selected = set()

    def can_add(idx, selected):
        for conflict in conflicts[idx]:
            if conflict[0] in selected and conflict[1] in selected and conflict[2] in selected:
                return False
        return True

    # Phase 1: Multiple random greedy constructions
    # Use different random seeds for diversity
    for seed in range(200):
        if time.time() - start_time > time_limit - 30:
            break

        current_order = list(range(total_edges))
        rng.seed(seed)
        rng.shuffle(current_order)

        selected = set()
        count = 0

        for idx in current_order:
            if can_add(idx, selected):
                selected.add(idx)
                count += 1

        if count > best_count:
            best_count = count
            best_selected = selected.copy()

        if best_count >= 308:
            break

    # Phase 2: Local search improvement from the best solution found
    # Try to add edges by removing conflicting edges (1-for-k swaps where k < 1 is impossible,
    # so we look for 1-for-1 or 1-for-2 swaps that might lead to further improvements)
    
    if best_count > 0:
        selected = best_selected.copy()
        count = best_count
        improved = True
        max_iterations = 50
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit - 5:
                break
            
            if not improved:
                break
            improved = False
            
            # Build list of unselected edges
            unselected = [i for i in range(total_edges) if i not in selected]
            rng.shuffle(unselected)
            
            # Try to add each unselected edge
            for idx in unselected:
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Find all conflicts that would be completed by adding idx
                conflicting_edges = set()
                for conflict in conflicts[idx]:
                    if conflict[0] in selected and conflict[1] in selected and conflict[2] in selected:
                        conflicting_edges.add(conflict[0])
                        conflicting_edges.add(conflict[1])
                        conflicting_edges.add(conflict[2])
                
                if not conflicting_edges:
                    # Can add directly
                    selected.add(idx)
                    count += 1
                    improved = True
                    continue
                
                # Try removing one edge from conflicting_edges to see if idx can be added
                for rem in list(conflicting_edges):
                    new_selected = selected - {rem}
                    new_selected.add(idx)
                    
                    # Check if idx can be added to new_selected
                    can = True
                    for conflict in conflicts[idx]:
                        if all(k in new_selected for k in conflict):
                            can = False
                            break
                    
                    if can:
                        # Check that removing rem and adding idx doesn't create new C4s
                        # (removing edges can't create C4s, adding idx we already checked)
                        # But we need to make sure no other edge now has a completed conflict
                        # Actually, removing an edge can only help, so this is safe
                        selected = new_selected
                        # count stays the same (remove 1, add 1)
                        improved = True
                        break
                
                if improved:
                    break
            
            # Also try: for each selected edge, see if removing it allows adding 2+ edges
            # This is more expensive, so do it less frequently
            if iteration % 5 == 4:
                selected_edges_list = list(selected)
                rng.shuffle(selected_edges_list)
                
                for rem in selected_edges_list[:50]:  # Try first 50
                    if time.time() - start_time > time_limit - 5:
                        break
                    
                    new_selected = selected - {rem}
                    new_count = count - 1
                    new_improved = False
                    
                    # Try to add edges that were previously blocked by rem
                    for add_idx in conflict_partners[rem]:
                        if add_idx in new_selected:
                            continue
                        if can_add(add_idx, new_selected):
                            new_selected.add(add_idx)
                            new_count += 1
                            new_improved = True
                    
                    if new_count > count:
                        selected = new_selected
                        count = new_count
                        improved = True
                        break

        if count > best_count:
            best_count = count
            best_selected = selected.copy()

    # Phase 3: Simulated annealing for further improvement
    if best_count > 0 and time.time() - start_time < time_limit - 10:
        selected = best_selected.copy()
        count = best_count
        best_sa_count = count
        best_sa_selected = selected.copy()
        
        T = 1.0
        alpha = 0.999999  # Very slow cooling
        min_T = 0.001
        
        sa_start = time.time()
        sa_time_limit = min(30, time_limit - (sa_start - start_time) - 5)
        
        while time.time() - sa_start < sa_time_limit and T > min_T:
            # Pick a random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a random unselected edge
                unselected = [i for i in range(total_edges) if i not in selected]
                if not unselected:
                    T *= alpha
                    continue
                idx = rng.choice(unselected)
                
                # Check if it can be added
                if can_add(idx, selected):
                    selected.add(idx)
                    count += 1
                    delta = 1
                else:
                    # Find how many edges need to be removed
                    conflicting_edges = set()
                    for conflict in conflicts[idx]:
                        if conflict[0] in selected and conflict[1] in selected and conflict[2] in selected:
                            conflicting_edges.add(conflict[0])
                            conflicting_edges.add(conflict[1])
                            conflicting_edges.add(conflict[2])
                    
                    if len(conflicting_edges) <= 2:
                        # Try removing a random subset of conflicting edges
                        rem_set = set(rng.sample(list(conflicting_edges), min(len(conflicting_edges), 1)))
                        new_selected = selected - rem_set
                        new_selected.add(idx)
                        
                        can = True
                        for conflict in conflicts[idx]:
                            if all(k in new_selected for k in conflict):
                                can = False
                                break
                        
                        if can:
                            delta = 1 - len(rem_set)
                            selected = new_selected
                            count += delta
                        else:
                            delta = -10  # Penalty for invalid move
                    else:
                        delta = -10
                        continue
            elif move_type < 0.7:
                # Try to remove a random selected edge
                if not selected:
                    T *= alpha
                    continue
                idx = rng.choice(list(selected))
                selected.remove(idx)
                count -= 1
                delta = -1
            else:
                # Swap: remove one selected, add one unselected
                if not selected:
                    T *= alpha
                    continue
                unselected = [i for i in range(total_edges) if i not in selected]
                if not unselected:
                    T *= alpha
                    continue
                
                rem = rng.choice(list(selected))
                add = rng.choice(unselected)
                
                new_selected = selected - {rem}
                new_selected.add(add)
                
                can = True
                for conflict in conflicts[add]:
                    if all(k in new_selected for k in conflict):
                        can = False
                        break
                
                if can:
                    delta = 0  # Neutral swap
                    selected = new_selected
                else:
                    delta = -5
                    continue
            
            # Accept or reject based on Metropolis criterion
            if delta >= 0:
                if count > best_sa_count:
                    best_sa_count = count
                    best_sa_selected = selected.copy()
            else:
                # Accept with probability exp(delta / T)
                if T > 0:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        pass  # Already applied
                    else:
                        # Revert the move
                        if move_type < 0.4:
                            if delta == 1:
                                selected.remove(add)
                                count -= 1
                            elif delta < 0:
                                # Need to revert the removal and addition
                                # This is complex, so just skip
                                pass
                        elif move_type < 0.7:
                            selected.add(rem)
                            count += 1
                        else:
                            # Revert swap
                            new_selected2 = selected - {add}
                            new_selected2.add(rem)
                            selected = new_selected2
            
            T *= alpha
        
        if best_sa_count > best_count:
            best_count = best_sa_count
            best_selected = best_sa_selected.copy()

    # Convert best_selected to list of edges
    result = []
    for idx in best_selected:
        u, v = edges[idx]
        result.append([u, v])
    
    return result
