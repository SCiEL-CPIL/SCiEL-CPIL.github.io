# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    """
    p = 7
    k = 3  # alpha^2 = 3
    q = p * p  # 49
    
    def add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1 * a2 + b1 * b2 * k) % p, (a1 * b2 + a2 * b1) % p)
    
    def neg(e):
        return ((-e[0]) % p, (-e[1]) % p)
    
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(field_elements)}
    
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i, e1 in enumerate(field_elements):
        for j, e2 in enumerate(field_elements):
            add_table[i, j] = elem_to_idx[add(e1, e2)]
            mul_table[i, j] = elem_to_idx[mul(e1, e2)]
    
    total_points = q * q  # 2401
    num_dirs = q + 1  # 50
    
    lines = [dict() for _ in range(num_dirs)]
    point_line = np.zeros((total_points, num_dirs), dtype=np.int64)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    x_idx = elem_to_idx[x]
                    y_idx = elem_to_idx[y]
                    pt_idx = x_idx * q + y_idx
                    
                    lines[q][x_idx] = lines[q].get(x_idx, [])
                    lines[q][x_idx].append(pt_idx)
                    point_line[pt_idx, q] = x_idx
                    
                    for m_idx, m in enumerate(field_elements):
                        mx_idx = mul_table[m_idx, x_idx]
                        neg_mx_idx = elem_to_idx[neg(field_elements[mx_idx])]
                        line_id = add_table[y_idx, neg_mx_idx]
                        lines[m_idx][line_id] = lines[m_idx].get(line_id, [])
                        lines[m_idx][line_id].append(pt_idx)
                        point_line[pt_idx, m_idx] = line_id
    
    for d in range(num_dirs):
        for lid in lines[d]:
            lines[d][lid] = np.array(lines[d][lid], dtype=np.int64)
    
    qrs = {0, 1, 2, 4}
    in_N = np.ones(total_points, dtype=bool)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = elem_to_idx[(xa, xb)]
                    y_idx = elem_to_idx[(ya, yb)]
                    pt_idx = x_idx * q + y_idx
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in qrs:
                            in_N[pt_idx] = False
    
    line_N_count = np.zeros((num_dirs, q), dtype=np.int64)
    for d in range(num_dirs):
        for lid, pts in lines[d].items():
            line_N_count[d, lid] = np.sum(in_N[pts])
    
    def is_witnessed_fast(pt_idx):
        for d in range(num_dirs):
            lid = point_line[pt_idx, d]
            if line_N_count[d, lid] == q - 1:
                return True
        return False
    
    start_time = time.time()
    timeout = 550
    
    # For efficient checking, precompute for each point the list of (d, lid) it's on
    # And for each (d, lid), the list of points
    
    # Try to exclude points greedily
    # Key optimization: when we exclude pt, only points on lines through pt can be affected
    # And a point p2 is affected only if it was witnessed by a line through pt
    
    excluded = 0
    max_rounds = 30
    
    for rnd in range(max_rounds):
        if time.time() - start_time > timeout:
            break
        
        progress = False
        for pt_idx in range(total_points):
            if time.time() - start_time > timeout:
                break
            if not in_N[pt_idx]:
                continue
            
            # Try excluding pt_idx
            in_N[pt_idx] = False
            for d in range(num_dirs):
                lid = point_line[pt_idx, d]
                line_N_count[d, lid] -= 1
            
            # Check pt_idx is still witnessed
            if not is_witnessed_fast(pt_idx):
                in_N[pt_idx] = True
                for d in range(num_dirs):
                    lid = point_line[pt_idx, d]
                    line_N_count[d, lid] += 1
                continue
            
            # Check affected points
            # A point p2 is affected if p2 is on a line through pt_idx
            # AND p2 was witnessed by that line (i.e., before exclusion, line_N_count was q-1 for that line from p2's perspective)
            # Actually, p2 is witnessed by line L if all other points on L are in N
            # Before excluding pt_idx: if pt_idx was in N, and all other points were in N, then p2 was witnessed by L
            # After excluding pt_idx: p2 is witnessed by L iff all points on L except p2 are in N (which now includes checking pt_idx is not needed)
            # Wait, no: p2 is witnessed by L if all points on L except p2 are in N
            # Before: this included pt_idx (which was in N)
            # After: pt_idx is not in N, so if pt_idx != p2, then p2 is NOT witnessed by L anymore (unless there's another direction)
            
            # So we need to check: for each p2 on lines through pt_idx, is p2 still witnessed by some direction?
            
            valid = True
            checked = set()
            for d in range(num_dirs):
                lid = point_line[pt_idx, d]
                for p2 in lines[d][lid]:
                    if p2 == pt_idx or p2 in checked:
                        continue
                    checked.add(p2)
                    if not is_witnessed_fast(p2):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                progress = True
                excluded += 1
            else:
                in_N[pt_idx] = True
                for d in range(num_dirs):
                    lid = point_line[pt_idx, d]
                    line_N_count[d, lid] += 1
        
        if not progress:
            break
    
    points = []
    for pt_idx in range(total_points):
        if in_N[pt_idx]:
            x_idx = pt_idx // q
            y_idx = pt_idx % q
            xa = x_idx // p
            xb = x_idx % p
            ya = y_idx // p
            yb = y_idx % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
