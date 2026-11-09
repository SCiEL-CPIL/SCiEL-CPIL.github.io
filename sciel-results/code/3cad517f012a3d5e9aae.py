"""Model-and-optimize approach for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Module-level state
_dm_cache = {}
_start_time = None
_instance_id = 0

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    global _instance_id
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
        _instance_id += 1
        _start_time = time.time()
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > 140

def _greedy_complete_route(start_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Fast greedy heuristic to complete a route from start_node.
    Uses nearest-neighbor with savings-based scoring.
    Returns the total distance of the completed route (from start_node to depot).
    """
    total_dist = 0.0
    current = start_node
    remaining_cap = rest_capacity
    remaining = list(unvisited)
    
    while remaining:
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            feas_arr = np.array(feasible)
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            dist_current_depot = dm[current, depot]
            
            # Savings: benefit of visiting candidate vs going back to depot
            savings = dist_current_depot + dists_to_depot - dists_to_current
            
            # Normalize savings to [0, 1]
            s_min = np.min(savings)
            s_max = np.max(savings)
            if s_max > s_min:
                norm_savings = (savings - s_min) / (s_max - s_min)
            else:
                norm_savings = np.zeros_like(savings)
            
            # Geometric alignment in route completion
            if dist_current_depot > 0:
                align = (dist_current_depot - dists_to_depot) / dist_current_depot
            else:
                align = np.zeros_like(dists_to_depot)
            al_min = np.min(align)
            al_max = np.max(align)
            if al_max > al_min:
                norm_align = (align - al_min) / (al_max - al_min)
            else:
                norm_align = np.zeros_like(align)
            
            # Combined score: minimize distance, reward savings and alignment
            scores = dists_to_current + 0.08 * dists_to_depot - 0.30 * norm_savings * dist_current_depot - 0.15 * norm_align * dist_current_depot
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
        
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
    
    total_dist += dm[current, depot]
    return total_dist

def _improved_greedy_complete_route(start_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Improved greedy that looks ahead 1-2 steps.
    Uses savings-based scoring with 1-step lookahead for small sets.
    """
    total_dist = 0.0
    current = start_node
    remaining_cap = rest_capacity
    remaining = list(unvisited)
    
    while remaining:
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        elif len(feasible) <= 8:
            # For small sets, do 1-step lookahead
            best_cost = float('inf')
            best_node = feasible[0]
            for cand in feasible:
                sim_remaining = [n for n in remaining if n != cand]
                sim_cap = remaining_cap - demands[cand]
                if sim_cap < 0:
                    continue
                cost = dm[current, cand] + _greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
                if cost < best_cost:
                    best_cost = cost
                    best_node = cand
            next_node = best_node
        else:
            # For larger sets, use faster heuristic with savings component
            feas_arr = np.array(feasible)
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            dist_current_depot = dm[current, depot]
            
            # Savings: d(current,depot) + d(depot,cand) - d(current,cand)
            savings = dist_current_depot + dists_to_depot - dists_to_current
            
            # Normalize savings to [0, 1] for stable weighting
            s_min = np.min(savings)
            s_max = np.max(savings)
            if s_max > s_min:
                norm_savings = (savings - s_min) / (s_max - s_min)
            else:
                norm_savings = np.zeros_like(savings)
            
            # Geometric alignment in improved route completion
            if dist_current_depot > 0:
                align = (dist_current_depot - dists_to_depot) / dist_current_depot
            else:
                align = np.zeros_like(dists_to_depot)
            al_min = np.min(align)
            al_max = np.max(align)
            if al_max > al_min:
                norm_align = (align - al_min) / (al_max - al_min)
            else:
                norm_align = np.zeros_like(align)
            
            # Combined score: minimize distance, penalize far from depot, reward savings and alignment
            scores = dists_to_current + 0.08 * dists_to_depot - 0.30 * norm_savings * dist_current_depot - 0.15 * norm_align * dist_current_depot
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
        
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
    
    total_dist += dm[current, depot]
    return total_dist

def _savings_based_select(current_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Clarke-Wright savings-inspired selection with geometric alignment bias.
    """
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])
    
    feas_arr = np.array(feasible)
    dist_current = dm[current_node, feas_arr]
    dist_depot = dm[feas_arr, depot]
    dist_current_depot = dm[current_node, depot]
    
    # Savings of visiting candidate instead of going back to depot
    savings = dist_current_depot + dist_depot - dist_current
    
    # Normalize savings
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max > s_min:
        norm_savings = (savings - s_min) / (s_max - s_min)
    else:
        norm_savings = np.zeros_like(savings)
    
    # Geometric alignment: prefer nodes that reduce distance-to-depot (descent direction)
    # alignment = (dist_current_depot - dist_depot) / dist_current_depot
    # Positive means candidate is closer to depot than current node is
    if dist_current_depot > 0:
        alignment = (dist_current_depot - dist_depot) / dist_current_depot
    else:
        alignment = np.zeros_like(dist_depot)
    
    # Normalize alignment to [0, 1]
    a_min = np.min(alignment)
    a_max = np.max(alignment)
    if a_max > a_min:
        norm_alignment = (alignment - a_min) / (a_max - a_min)
    else:
        norm_alignment = np.zeros_like(alignment)
    
    # Combined score: distance + depot penalty - savings reward - alignment reward
    score = dist_current + 0.08 * dist_depot - 0.30 * norm_savings * dist_current_depot - 0.15 * norm_alignment * dist_current_depot
    
    best_idx = np.argmin(score)
    return int(feas_arr[best_idx])

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: Hybrid of savings-based filtering and lookahead with soft-min smoothing.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    n_feasible = len(feasible)
    
    # If few feasible nodes, do full lookahead
    if n_feasible <= 8:
        best_cost = float('inf')
        best_node = feasible[0]
        
        for cand in feasible:
            sim_remaining = [n for n in unvisited if n != cand]
            sim_cap = rest_capacity - demands[cand]
            if sim_cap < 0:
                continue
            sim_cost = _improved_greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
            total_cost = dm[current_node, cand] + sim_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_node = cand
        
        return int(best_node)
    else:
        # Hybrid: use savings-based scoring to filter top-k, then lookahead
        feas_arr = np.array(feasible)
        dists = dm[current_node, feas_arr]
        dists_depot = dm[feas_arr, depot]
        dist_cd = dm[current_node, depot]
        
        # Savings score
        savings = dist_cd + dists_depot - dists
        
        # Normalize savings
        s_min = np.min(savings)
        s_max = np.max(savings)
        if s_max > s_min:
            norm_savings = (savings - s_min) / (s_max - s_min)
        else:
            norm_savings = np.zeros_like(savings)
        
        # Geometric alignment bias
        if dist_cd > 0:
            alignment = (dist_cd - dists_depot) / dist_cd
        else:
            alignment = np.zeros_like(dists_depot)
        a_min = np.min(alignment)
        a_max = np.max(alignment)
        if a_max > a_min:
            norm_alignment = (alignment - a_min) / (a_max - a_min)
        else:
            norm_alignment = np.zeros_like(alignment)
        
        # Combined score for initial filtering
        score = dists + 0.08 * dists_depot - 0.30 * norm_savings * dist_cd - 0.15 * norm_alignment * dist_cd
        
        # Get top-k closest by combined score
        k = min(8, n_feasible)
        top_k_indices = np.argpartition(score, k - 1)[:k]
        top_k_nodes = [int(feas_arr[i]) for i in top_k_indices]
        
        best_cost = float('inf')
        best_node = top_k_nodes[0]
        
        for cand in top_k_nodes:
            sim_remaining = [n for n in unvisited if n != cand]
            sim_cap = rest_capacity - demands[cand]
            if sim_cap < 0:
                continue
            sim_cost = _improved_greedy_complete_route(cand, depot, sim_remaining, sim_cap, demands, dm)
            total_cost = dm[current_node, cand] + sim_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_node = cand
        
        return int(best_node)
# EVOLVE-BLOCK-END
