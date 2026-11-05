"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        # Generate edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def min_vertex_cover(n, edges):
        if not edges:
            return []
        
        # Identify non-isolated vertices
        vertices_involved = set()
        for u, v in edges:
            vertices_involved.add(u)
            vertices_involved.add(v)
        
        if not vertices_involved:
            return []
        
        # Map involved vertices to indices 0..m-1
        verts = sorted(vertices_involved)
        v_to_idx = {v: i for i, v in enumerate(verts)}
        m = len(verts)
        
        # Build constraint matrix for edges: x_u + x_v >= 1
        # Rows: edges, Cols: vertices
        rows = []
        cols = []
        vals = []
        
        for u, v in edges:
            iu = v_to_idx[u]
            iv = v_to_idx[v]
            rows.append(len(rows))
            cols.append(iu)
            vals.append(1)
            
            rows.append(len(rows))
            cols.append(iv)
            vals.append(1)
            
        A = csr_matrix((vals, (rows, cols)), shape=(len(edges), m))
        
        # Objective: minimize sum(x)
        c = np.ones(m)
        
        # Constraints: A @ x >= 1
        # LinearConstraint(A, lb=np.ones(len(edges)), ub=np.inf)
        # Note: milp expects constraints in form lb <= A @ x <= ub
        constraint = LinearConstraint(A, lb=np.ones(len(edges)), ub=np.inf)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(m), ub=np.ones(m))
        
        # Integrality: all variables are integers
        integrality = np.ones(m)
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                # Extract solution
                solution = res.x
                # Due to floating point, threshold at 0.5
                selected_indices = [i for i in range(m) if solution[i] > 0.5]
                selected_vertices = [verts[i] for i in selected_indices]
                return sorted(selected_vertices)
            else:
                # Fallback: if ILP fails, use a greedy heuristic
                # Greedy: pick vertex covering most uncovered edges
                covered_edges = set()
                selected = []
                remaining_edges = list(edges)
                
                # Map edge to set of vertices for quick lookup? 
                # Simple greedy:
                while remaining_edges:
                    # Count occurrences of each vertex in remaining_edges
                    counts = {}
                    for u, v in remaining_edges:
                        counts[u] = counts.get(u, 0) + 1
                        counts[v] = counts.get(v, 0) + 1
                    
                    if not counts:
                        break
                        
                    # Pick vertex with max count
                    best_v = max(counts, key=counts.get)
                    selected.append(best_v)
                    
                    # Remove all edges incident to best_v
                    new_remaining = []
                    for u, v in remaining_edges:
                        if u != best_v and v != best_v:
                            new_remaining.append((u, v))
                    remaining_edges = new_remaining
                    
                return sorted(selected)
                
        except Exception:
            # Fallback to greedy on exception
            selected = []
            remaining_edges = list(edges)
            while remaining_edges:
                counts = {}
                for u, v in remaining_edges:
                    counts[u] = counts.get(u, 0) + 1
                    counts[v] = counts.get(v, 0) + 1
                if not counts:
                    break
                best_v = max(counts, key=counts.get)
                selected.append(best_v)
                remaining_edges = [(u, v) for u, v in remaining_edges if u != best_v and v != best_v]
            return sorted(selected)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # If we are running out of time, return what we have (empty lists for remaining)
            # This is a safety net, should not trigger for these small graphs
            out.append([])
            continue
            
        edges = instance(n, p, seed)
        cover = min_vertex_cover(n, edges)
        out.append(cover)
        
    return out
# EVOLVE-BLOCK-END
