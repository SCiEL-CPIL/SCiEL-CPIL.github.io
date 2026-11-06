import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def construct():
    n = 20
    num_colors = 20
    
    # Edge indices
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_idx = {e: i for i, e in enumerate(edges)}
    m_edges = len(edges)
    
    # Variables:
    # 0 .. m_edges-1 : x_uv (edge exists)
    # m_edges .. m_edges + m_edges*num_colors - 1 : y_uv_c (edge uv has color c)
    
    num_vars = m_edges + m_edges * num_colors
    
    # Objective: Maximize sum x_uv
    # milp minimizes, so minimize -sum x_uv
    c_obj = np.zeros(num_vars)
    for i in range(m_edges):
        c_obj[i] = -1.0
        
    # Constraints:
    # 1. y_uv_c <= x_uv
    # 2. sum_c y_uv_c == x_uv
    # 3. For each 4-cycle C and color c, sum_{e in C} y_e_c <= 1
    
    A_rows = []
    A_cols = []
    A_data = []
    
    b_lower = []
    b_upper = []
    
    def add_constraint(row, lower, upper):
        """
        row: dict {var_idx: coeff}
        """
        for var, coeff in row.items():
            A_rows.append(len(A_data))
            A_cols.append(var)
            A_data.append(coeff)
        b_lower.append(lower)
        b_upper.append(upper)
        
    # Constraint 1 & 2: Linking
    # For each edge e:
    # sum_c y_e_c - x_e == 0
    # y_e_c - x_e <= 0 (implied by sum==0 and y>=0? No, y can be 1, x 0 -> sum 1 != 0. 
    # Actually, if sum y = x, and y in {0,1}, x in {0,1}, then if x=0, all y=0. If x=1, exactly one y=1.
    # So just sum_c y_e_c - x_e == 0 is sufficient if bounds are 0..1.
    
    for e_idx, (u, v) in enumerate(edges):
        row = {}
        row[e_idx] = -1.0 # -x_e
        for c in range(num_colors):
            y_idx = m_edges + e_idx * num_colors + c
            row[y_idx] = 1.0 # +y_e_c
        add_constraint(row, 0.0, 0.0)
        
    # Constraint 3: Rainbow 4-cycles
    # Generate all 4-cycles
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a specific ordering?
    # No, the constraint is on the *subgraph*.
    # "Every 4-cycle (as a subgraph) must receive four distinct colours"
    # A 4-cycle subgraph is defined by 4 vertices and the 4 edges connecting them in a cycle.
    # In a simple graph, a set of 4 vertices can form at most one 4-cycle if it's a cycle?
    # No, $K_4$ has 3 distinct 4-cycles (Hamiltonian cycles).
    # The constraint applies to *every* 4-cycle subgraph.
    # So for each set of 4 vertices $\{v_1, v_2, v_3, v_4\}$, there are 3 distinct 4-cycles.
    # Cycle 1: v1-v2-v3-v4-v1 (Edges: (v1,v2), (v2,v3), (v3,v4), (v4,v1))
    # Cycle 2: v1-v2-v4-v3-v1 (Edges: (v1,v2), (v2,v4), (v4,v3), (v3,v1))
    # Cycle 3: v1-v3-v2-v4-v1 (Edges: (v1,v3), (v3,v2), (v2,v4), (v4,v1))
    
    # We need to add constraints for all these.
    # For each such cycle (set of 4 edges) and each color c:
    # sum_{e in cycle} y_e_c <= 1
    
    vertices = list(range(n))
    cycle_count = 0
    for v1, v2, v3, v4 in itertools.combinations(vertices, 4):
        # Define the 3 cycles
        # Note: edge indices are unordered pairs
        e1 = (min(v1,v2), max(v1,v2))
        e2 = (min(v2,v3), max(v2,v3))
        e3 = (min(v3,v4), max(v3,v4))
        e4 = (min(v4,v1), max(v4,v1))
        
        # Cycle 1: v1-v2-v3-v4-v1
        edges_c1 = [edge_idx[e1], edge_idx[e2], edge_idx[e3], edge_idx[e4]]
        
        # Cycle 2: v1-v2-v4-v3-v1
        e2_alt = (min(v2,v4), max(v2,v4))
        e3_alt = (min(v4,v3), max(v4,v3))
        e4_alt = (min(v3,v1), max(v3,v1))
        edges_c2 = [edge_idx[e1], edge_idx[e2_alt], edge_idx[e3_alt], edge_idx[e4_alt]]
        
        # Cycle 3: v1-v3-v2-v4-v1
        e1_alt = (min(v1,v3), max(v1,v3))
        e2_alt2 = (min(v3,v2), max(v3,v2)) # same as e2
        e3_alt2 = (min(v2,v4), max(v2,v4)) # same as e2_alt
        e4_alt2 = (min(v4,v1), max(v4,v1)) # same as e4
        edges_c3 = [edge_idx[e1_alt], edge_idx[e2], edge_idx[e2_alt], edge_idx[e4]]
        
        cycles = [edges_c1, edges_c2, edges_c3]
        
        for cycle in cycles:
            for c in range(num_colors):
                row = {}
                for e_i in cycle:
                    y_idx = m_edges + e_i * num_colors + c
                    row[y_idx] = 1.0
                add_constraint(row, 0.0, 1.0)
                cycle_count += 1

    A = csr_matrix((A_data, (A_rows, A_cols)), shape=(len(b_lower), num_vars))
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=np.ones(num_vars), bounds=bounds, options={"time_limit": 110})
        
        if res.status == 0 or res.status == 1: # Optimal or Time Limit
            x = res.x
            # Extract edges
            final_edges = []
            for i, (u, v) in enumerate(edges):
                if x[i] > 0.5:
                    # Find color
                    for c in range(num_colors):
                        y_idx = m_edges + i * num_colors + c
                        if x[y_idx] > 0.5:
                            final_edges.append([u, v, c])
                            break
            return final_edges
        else:
            # Fallback to empty or simple
            return []
    except Exception:
        return []
