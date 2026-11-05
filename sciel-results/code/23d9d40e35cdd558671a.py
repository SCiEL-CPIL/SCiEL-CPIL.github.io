"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d
    total_edges = d * (1 << (d - 1))  # 448
    
    # Precompute all edges and their face memberships
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y, i))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    # For each edge, precompute the set of faces it belongs to
    # A face is determined by two coordinates (i, j) and two vertices that differ in those bits
    # Edge (x, y) with direction i is in faces (i, j) for all j != i
    # The face is identified by (i, j, base_vertex) where base_vertex has bits i and j cleared
    
    # For fast conflict checking: for each edge, store the list of (face_id, other_edges_in_face)
    # Actually, let's use a different approach: for each pair of edges that share a face, 
    # track conflicts. But that's complex.
    
    # Simpler: for each edge e with direction i, the faces it's in are (i, j) for j != i.
    # In face (i, j), the four edges are:
    #   (x, x^i), (x^j, x^i^j), (x, x^j), (x^i, x^i^j)
    # where x is the vertex with bits i and j cleared.
    
    # Let's precompute for each edge the list of "potential conflicts":
    # When we add edge e, we need to check if any face containing e already has 3 other edges.
    # So for each face containing e, check if 3 of the other 3 edges are already selected.
    
    # Precompute face membership:
    # face_members[face_id] = list of 4 edge indices
    # edge_faces[edge_idx] = list of face_ids containing this edge
    
    face_members = {}
    edge_faces = [[] for _ in range(total_edges)]
    
    for x in range(n):
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                # Face defined by coordinates i and j, base vertex x with bits i,j cleared
                base = x & ~(1 << i) & ~(1 << j)
                if base != x:
                    continue  # Only count once when x is the base
                # Four vertices of this face
                v0 = base
                v1 = base | (1 << i)
                v2 = base | (1 << j)
                v3 = base | (1 << i) | (1 << j)
                
                # Four edges
                e0 = (v0, v1)  # direction i
                e1 = (v2, v3)  # direction i
                e2 = (v0, v2)  # direction j
                e3 = (v1, v3)  # direction j
                
                # Get edge indices
                idx0 = edge_to_idx.get((min(e0[0], e0[1]), max(e0[0], e0[1])))
                idx1 = edge_to_idx.get((min(e1[0], e1[1]), max(e1[0], e1[1])))
                idx2 = edge_to_idx.get((min(e2[0], e2[1]), max(e2[0], e2[1])))
                idx3 = edge_to_idx.get((min(e3[0], e3[1]), max(e3[0], e3[1])))
                
                if idx0 is None or idx1 is None or idx2 is None or idx3 is None:
                    continue
                
                face_id = (i, j, base)
                face_members[face_id] = [idx0, idx1, idx2, idx3]
                for idx in [idx0, idx1, idx2, idx3]:
                    edge_faces[idx].append(face_id)
    
    # Now implement simulated annealing / local search
    # State: set of selected edges
    # Moves: add an edge, remove an edge, swap (remove one, add another)
    
    def count_edges(selected):
        return len(selected)
    
    def is_valid(selected):
        """Check if no face has all 4 edges selected."""
        for face_id, members in face_members.items():
            if all(m in selected for m in members):
                return False
        return True
    
    def get_conflict_count(selected):
        """Count number of faces with all 4 edges selected."""
        count = 0
        for face_id, members in face_members.items():
            if all(m in selected for m in members):
                count += 1
        return count
    
    def try_add(selected, edge_idx):
        """Try to add edge_idx to selected. Return True if it doesn't create a conflict."""
        if edge_idx in selected:
            return True
        # Check all faces containing this edge
        for face_id in edge_faces[edge_idx]:
            members = face_members[face_id]
            # Count how many of the other 3 edges are already selected
            other_count = 0
            for m in members:
                if m != edge_idx and m in selected:
                    other_count += 1
            if other_count == 3:
                return False
        return True
    
    # Start with a greedy solution
    def greedy_init(order):
        selected = set()
        for edge_idx in order:
            if try_add(selected, edge_idx):
                selected.add(edge_idx)
        return selected
    
    # Generate a random order
    def random_order():
        order = list(range(total_edges))
        random.shuffle(order)
        return order
    
    # Local search: try to improve by removing edges from conflicting faces
    def local_search(selected, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Find all conflicting faces
            conflicts = []
            for face_id, members in face_members.items():
                if all(m in selected for m in members):
                    conflicts.append(members)
            
            if not conflicts:
                break
            
            # For each conflict, try removing one of its edges
            # Choose the edge whose removal frees the most potential additions
            for members in conflicts:
                if time.time() >= time_limit:
                    break
                for m in members:
                    if m not in selected:
                        continue
                    # Try removing m
                    new_selected = selected - {m}
                    # Count how many edges we could now add
                    gain = 0
                    for edge_idx in range(total_edges):
                        if edge_idx not in new_selected and edge_idx != m:
                            if try_add(new_selected, edge_idx):
                                gain += 1
                    if gain > 0:
                        # Actually just remove m and see if we can add others
                        selected = new_selected
                        improved = True
                        break
                if improved:
                    break
            
            # After removing some edges, try to add back as many as possible
            if improved:
                for edge_idx in range(total_edges):
                    if edge_idx not in selected:
                        if try_add(selected, edge_idx):
                            selected.add(edge_idx)
        
        return selected
    
    # Simulated annealing approach
    def simulated_annealing(selected, time_limit):
        current = set(selected)
        best = set(current)
        best_count = len(current)
        
        T = 50.0
        cooling = 0.999
        
        while time.time() < time_limit:
            T *= cooling
            if T < 0.01:
                T = 0.01
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Add a random edge
                edge_idx = random.randint(0, total_edges - 1)
                if edge_idx in current:
                    continue
                new_selected = current | {edge_idx}
                delta = 1 - get_conflict_count(new_selected) + get_conflict_count(current)
                # Actually, we want to track conflicts
                old_conflicts = get_conflict_count(current)
                new_conflicts = get_conflict_count(new_selected)
                delta = (len(new_selected) - len(current)) - 10 * (new_conflicts - old_conflicts)
            elif move_type < 0.8:
                # Remove a random edge
                if not current:
                    continue
                edge_idx = random.choice(list(current))
                new_selected = current - {edge_idx}
                old_conflicts = get_conflict_count(current)
                new_conflicts = get_conflict_count(new_selected)
                delta = (len(new_selected) - len(current)) - 10 * (new_conflicts - old_conflicts)
            else:
                # Swap: remove one, add another
                if not current:
                    continue
                remove_idx = random.choice(list(current))
                add_idx = random.randint(0, total_edges - 1)
                if add_idx in current:
                    continue
                new_selected = (current - {remove_idx}) | {add_idx}
                old_conflicts = get_conflict_count(current)
                new_conflicts = get_conflict_count(new_selected)
                delta = (len(new_selected) - len(current)) - 10 * (new_conflicts - old_conflicts)
            
            # Accept if better, or with probability exp(delta/T)
            if delta > 0 or random.random() < math.exp(delta / T):
                current = new_selected
                if len(current) > best_count:
                    best = set(current)
                    best_count = len(current)
        
        return best
    
    # Main: try multiple restarts with different strategies
    best_result = None
    best_count = 0
    
    # First, try deterministic greedy
    order = list(range(total_edges))
    selected = greedy_init(order)
    selected = local_search(selected, start_time + 10)
    if len(selected) > best_count:
        best_count = len(selected)
        best_result = set(selected)
    
    # Now try randomized restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # Random order greedy
        order = random_order()
        selected = greedy_init(order)
        
        # Local search
        selected = local_search(selected, start_time + 15)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = set(selected)
        
        # Try simulated annealing from this solution
        if time.time() < deadline - 10:
            sa_result = simulated_annealing(selected, deadline - 5)
            if len(sa_result) > best_count:
                best_count = len(sa_result)
                best_result = set(sa_result)
    
    # Convert back to list of [u, v]
    result = []
    for edge_idx in sorted(best_result):
        x, y, i = edges[edge_idx]
        result.append([x, y])
    
    return result
# EVOLVE-BLOCK-END
