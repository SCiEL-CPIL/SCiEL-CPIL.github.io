import random
import time
import itertools

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    E = len(all_edges)
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Precompute all 4-cycles
    cycles = []
    edge_to_cycles = [[] for _ in range(E)]
    
    vertices = range(n)
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        # The 3 possible 4-cycles on these vertices
        # 1: a-b, b-c, c-d, d-a
        e1 = edge_index.get((a, b))
        e2 = edge_index.get((b, c))
        e3 = edge_index.get((c, d))
        e4 = edge_index.get((d, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles)-1)
                
        # 2: a-b, b-d, d-c, c-a
        e1 = edge_index.get((a, b))
        e2 = edge_index.get((b, d))
        e3 = edge_index.get((d, c))
        e4 = edge_index.get((c, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles)-1)
                
        # 3: a-c, c-b, b-d, d-a
        e1 = edge_index.get((a, c))
        e2 = edge_index.get((c, b))
        e3 = edge_index.get((b, d))
        e4 = edge_index.get((d, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles)-1)
                
    def count_violations(edge_colors):
        """Count number of 4-cycles that are not rainbow."""
        violations = 0
        for cyc in cycles:
            e1, e2, e3, e4 = cyc
            if edge_colors[e1] == -1 or edge_colors[e2] == -1 or edge_colors[e3] == -1 or edge_colors[e4] == -1:
                continue
            if edge_colors[e1] == edge_colors[e2] or edge_colors[e1] == edge_colors[e3] or edge_colors[e1] == edge_colors[e4] or \
               edge_colors[e2] == edge_colors[e3] or edge_colors[e2] == edge_colors[e4] or edge_colors[e3] == edge_colors[e4]:
                violations += 1
        return violations
    
    def try_add_edge(edge_colors, e_idx):
        """Try to add edge e_idx with a valid color. Returns True if added."""
        forbidden_colors = set()
        impossible = False
        
        for c_idx in edge_to_cycles[e_idx]:
            cycle = cycles[c_idx]
            other_edges = [e for e in cycle if e != e_idx]
            c1, c2, c3 = other_edges
            
            col1 = edge_colors[c1]
            col2 = edge_colors[c2]
            col3 = edge_colors[c3]
            
            if col1 == -1 or col2 == -1 or col3 == -1:
                continue
                
            if col1 == col2 or col2 == col3 or col1 == col3:
                impossible = True
                break
                
            forbidden_colors.add(col1)
            forbidden_colors.add(col2)
            forbidden_colors.add(col3)
            
        if impossible:
            return False
            
        if len(forbidden_colors) >= num_colors:
            return False
            
        allowed = [c for c in range(num_colors) if c not in forbidden_colors]
        if not allowed:
            return False
            
        edge_colors[e_idx] = random.choice(allowed)
        return True
    
    def local_search(edge_colors, max_iters=5000):
        """Perform local search: recolor edges to fix violations, then try adding edges."""
        current_violations = count_violations(edge_colors)
        best_violations = current_violations
        best_edge_count = sum(1 for c in edge_colors if c != -1)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Step 1: Try to recolor an edge to reduce violations
            if current_violations > 0:
                # Find an edge involved in a violation
                violated_edges = set()
                for cyc in cycles:
                    e1, e2, e3, e4 = cyc
                    if edge_colors[e1] == -1 or edge_colors[e2] == -1 or edge_colors[e3] == -1 or edge_colors[e4] == -1:
                        continue
                    if edge_colors[e1] == edge_colors[e2] or edge_colors[e1] == edge_colors[e3] or edge_colors[e1] == edge_colors[e4] or \
                       edge_colors[e2] == edge_colors[e3] or edge_colors[e2] == edge_colors[e4] or edge_colors[e3] == edge_colors[e4]:
                        violated_edges.update([e1, e2, e3, e4])
                
                if violated_edges:
                    e_idx = random.choice(list(violated_edges))
                    old_color = edge_colors[e_idx]
                    
                    # Try each color
                    for new_color in range(num_colors):
                        if new_color == old_color:
                            continue
                        edge_colors[e_idx] = new_color
                        new_violations = count_violations(edge_colors)
                        if new_violations < current_violations:
                            current_violations = new_violations
                            if current_violations < best_violations:
                                best_violations = current_violations
                                best_edge_count = sum(1 for c in edge_colors if c != -1)
                            break
                        else:
                            edge_colors[e_idx] = old_color
            else:
                # No violations, try to add edges
                improved = False
                for e_idx in range(E):
                    if edge_colors[e_idx] == -1:
                        if try_add_edge(edge_colors, e_idx):
                            improved = True
                            best_edge_count = sum(1 for c in edge_colors if c != -1)
                if not improved:
                    break
        
        return edge_colors, best_edge_count
    
    best_graph = None
    best_edge_count = 0
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Start from complete graph with random coloring
        edge_colors = [random.randint(0, num_colors - 1) for _ in range(E)]
        
        # Local search
        edge_colors, edge_count = local_search(edge_colors, max_iters=3000)
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_graph = []
            for i in range(E):
                if edge_colors[i] != -1:
                    u, v = all_edges[i]
                    best_graph.append([u, v, edge_colors[i]])
            
            # If we have a valid solution, try to improve it further
            if best_violations == 0:
                # Try to add more edges
                for e_idx in range(E):
                    if edge_colors[e_idx] == -1:
                        if try_add_edge(edge_colors, e_idx):
                            best_edge_count = sum(1 for c in edge_colors if c != -1)
                            best_graph = []
                            for i in range(E):
                                if edge_colors[i] != -1:
                                    u, v = all_edges[i]
                                    best_graph.append([u, v, edge_colors[i]])
    
    if best_graph is None:
        return []
        
    return best_graph

# EVOLVE-BLOCK-START
# (The function above is the implementation)
# EVOLVE-BLOCK-END
