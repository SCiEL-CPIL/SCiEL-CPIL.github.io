"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    total_edges = len(all_edges)
    
    # Precompute all 4-cycles in the complete graph
    # A 4-cycle is u-v-w-x-u where all four vertices are distinct
    # We represent each 4-cycle as a frozenset of its 4 edges
    cycles = []
    cycles_set = set()
    
    # For each set of 4 vertices, there are 3 distinct 4-cycles
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Three distinct 4-cycles on {a,b,c,d}:
        # 1: a-b-c-d-a
        # 2: a-b-d-c-a
        # 3: a-c-b-d-a
        for (v1, v2, v3, v4) in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
            edges_in_cycle = frozenset([
                (min(v1,v2), max(v1,v2)),
                (min(v2,v3), max(v2,v3)),
                (min(v3,v4), max(v3,v4)),
                (min(v4,v1), max(v4,v1))
            ])
            if edges_in_cycle not in cycles_set:
                cycles_set.add(edges_in_cycle)
                cycles.append(edges_in_cycle)
    
    num_cycles = len(cycles)
    
    # Map each edge to the list of cycle indices it belongs to
    edge_to_cycle_indices = {e: [] for e in all_edges}
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            if e in edge_to_cycle_indices:
                edge_to_cycle_indices[e].append(ci)
    
    # State representation:
    # edge_color: dict mapping (u,v) -> color (only for present edges)
    # edge_present: set of present edges
    
    def check_validity(edge_color, edge_present):
        """Check if the current coloring is valid (all 4-cycles are rainbow)."""
        for cycle in cycles:
            if all(e in edge_present for e in cycle):
                colors = [edge_color[e] for e in cycle]
                if len(set(colors)) < 4:
                    return False
        return True
    
    def count_invalid_cycles(edge_color, edge_present):
        """Count the number of invalid (non-rainbow) 4-cycles."""
        count = 0
        for cycle in cycles:
            if all(e in edge_present for e in cycle):
                colors = [edge_color[e] for e in cycle]
                if len(set(colors)) < 4:
                    count += 1
        return count
    
    def try_add_edge(edge, color, edge_color, edge_present, adj):
        """Try to add an edge with a given color. Return True if valid."""
        u, v = edge
        if edge in edge_present:
            return False
        
        # Check all cycles containing this edge
        for ci in edge_to_cycle_indices.get(edge, []):
            cycle = cycles[ci]
            # Check if all other 3 edges in this cycle are present
            other_edges = [e for e in cycle if e != edge]
            if all(e in edge_present for e in other_edges):
                # Check color conflict
                for e in other_edges:
                    if edge_color[e] == color:
                        return False
        return True
    
    def greedy_construct(random_seed=42):
        """Greedy construction with randomized edge ordering."""
        rng = random.Random(random_seed)
        edge_color = {}
        edge_present = set()
        adj = [set() for _ in range(n)]
        
        # Order edges: try to add high-degree edges first (more constrained)
        # Sort by number of cycles they participate in
        edge_order = sorted(all_edges, key=lambda e: len(edge_to_cycle_indices.get(e, [])), reverse=True)
        
        # Add some randomness
        for i in range(len(edge_order)):
            # Swap with a random earlier position to introduce diversity
            if i > 0 and rng.random() < 0.3:
                j = rng.randint(0, i)
                edge_order[i], edge_order[j] = edge_order[j], edge_order[i]
        
        for edge in edge_order:
            if time.time() > deadline - 5.0:
                break
            
            u, v = edge
            # Try colors in random order, prefer colors not yet used in adjacent edges
            colors = list(range(num_colors))
            rng.shuffle(colors)
            
            for color in colors:
                if try_add_edge(edge, color, edge_color, edge_present, adj):
                    edge_color[edge] = color
                    edge_present.add(edge)
                    adj[u].add(v)
                    adj[v].add(u)
                    break
        
        return edge_color, edge_present
    
    def local_search(edge_color, edge_present, max_iters=5000):
        """Local search: try to add more edges or swap colors to free up space."""
        rng = random.Random(12345)
        adj = [set() for _ in range(n)]
        for (u, v) in edge_present:
            adj[u].add(v)
            adj[v].add(u)
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            # Try to add a missing edge
            missing = [e for e in all_edges if e not in edge_present]
            if not missing:
                break
            
            # Pick a random missing edge
            edge = rng.choice(missing)
            u, v = edge
            
            # Try each color
            colors = list(range(num_colors))
            rng.shuffle(colors)
            
            added = False
            for color in colors:
                if try_add_edge(edge, color, edge_color, edge_present, adj):
                    edge_color[edge] = color
                    edge_present.add(edge)
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            
            if not added:
                # Try swapping colors of existing edges to make room
                # Find a cycle that's blocking this edge
                for ci in edge_to_cycle_indices.get(edge, []):
                    cycle = cycles[ci]
                    other_edges = [e for e in cycle if e != edge]
                    if all(e in edge_present for e in other_edges):
                        # Find color conflicts
                        for e in other_edges:
                            if e in edge_color:
                                old_color = edge_color[e]
                                # Try to find a new color for e
                                for new_color in range(num_colors):
                                    if new_color == old_color:
                                        continue
                                    # Check if new_color is valid for e
                                    valid = True
                                    for ci2 in edge_to_cycle_indices.get(e, []):
                                        cycle2 = cycles[ci2]
                                        for e2 in cycle2:
                                            if e2 != e and e2 in edge_present and edge_color.get(e2) == new_color:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        # Swap
                                        edge_color[e] = new_color
                                        # Now try to add the edge with old_color
                                        if try_add_edge(edge, old_color, edge_color, edge_present, adj):
                                            edge_color[edge] = old_color
                                            edge_present.add(edge)
                                            adj[u].add(v)
                                            adj[v].add(u)
                                            added = True
                                        # Revert if it didn't help
                                        if not added:
                                            edge_color[e] = old_color
                                        break
                            if added:
                                break
                    if added:
                        break
        
        return edge_color, edge_present
    
    # Main search: try multiple random seeds and combine with local search
    best_edge_count = 0
    best_state = {}
    
    # Phase 1: Multiple greedy constructions
    for seed in range(100):
        if time.time() > deadline - 10.0:
            break
        
        edge_color, edge_present = greedy_construct(seed)
        count = len(edge_present)
        
        if count > best_edge_count:
            best_edge_count = count
            best_state = dict(edge_color)
    
    # Phase 2: Local search from best states
    if best_state:
        if time.time() <= deadline - 10.0:
            edge_color, edge_present = local_search(dict(best_state), set(best_state.keys()), max_iters=10000)
            count = len(edge_present)
            if count > best_edge_count:
                best_edge_count = count
                best_state = dict(edge_color)
    
    # Phase 3: More aggressive local search with restarts
    for attempt in range(30):
        if time.time() > deadline - 5.0:
            break
        
        # Start from best state or a new greedy
        if attempt < 10 and best_state:
            edge_color = dict(best_state)
            edge_present = set(best_state.keys())
        else:
            edge_color, edge_present = greedy_construct(1000 + attempt)
        
        edge_color, edge_present = local_search(edge_color, edge_present, max_iters=5000)
        count = len(edge_present)
        
        if count > best_edge_count:
            best_edge_count = count
            best_state = dict(edge_color)
    
    # Final verification
    if best_state:
        edge_present = set(best_state.keys())
        if not check_validity(best_state, edge_present):
            # Remove invalid edges until valid
            for e in list(edge_present):
                # Check if removing this edge helps
                test_color = dict(best_state)
                test_present = set(edge_present)
                del test_color[e]
                test_present.discard(e)
                if check_validity(test_color, test_present):
                    del best_state[e]
                    edge_present.discard(e)
    
    # Convert to output format
    result = []
    for (u, v), color in best_state.items():
        result.append([u, v, color])
    
    return result
# EVOLVE-BLOCK-END
