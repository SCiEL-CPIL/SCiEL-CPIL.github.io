import random
import time
import math

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))  # 448
    
    # Generate all edges
    edges = []
    edge_to_id = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_id[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Precompute faces and edge-face incidence
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            others = [k for k in range(d) if k != i and k != j]
            for mask in range(2 ** len(others)):
                u = 0
                for idx, bit in enumerate(others):
                    if mask & (1 << idx):
                        u |= (1 << bit)
                
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_to_id[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_id[(min(v3, v4), max(v3, v4))]
                e3 = edge_to_id[(min(v1, v3), max(v1, v3))]
                e4 = edge_to_id[(min(v2, v4), max(v2, v4))]
                
                face_id = len(face_edges)
                face_edges.append([e1, e2, e3, e4])
                
                for e in [e1, e2, e3, e4]:
                    edge_faces[e].append(face_id)
    
    n_faces = len(face_edges)
    
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Initial solution: Random valid set via greedy
    # We want to start with a valid set to help SA.
    # Greedy: Start with all edges, remove edges that cause violations.
    selected = [True] * n_edges
    face_count = [4] * n_faces
    violated = set(range(n_faces))
    
    # Greedy removal
    while violated:
        # Count violations per edge
        edge_viol = [0] * n_edges
        for f in violated:
            for e in face_edges[f]:
                edge_viol[e] += 1
        
        if max(edge_viol) == 0:
            break
            
        # Remove edge with max violation
        # Randomize among max to allow different starts
        max_c = max(edge_viol)
        candidates = [e for e in range(n_edges) if edge_viol[e] == max_c]
        e_remove = random.choice(candidates)
        
        selected[e_remove] = False
        for f in edge_faces[e_remove]:
            face_count[f] -= 1
            if face_count[f] == 4:
                violated.add(f)
            else:
                violated.discard(f)
    
    current_score = sum(selected)
    current_violations = len(violated)
    best_score = current_score
    best_selected = selected.copy()
    best_face_count = face_count.copy()
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.9999
    
    # Penalty for violations
    PENALTY = 100
    
    def get_score(sel, viol_count):
        return sum(sel) - PENALTY * viol_count
    
    current_state_score = get_score(selected, current_violations)
    
    while time.time() < deadline:
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Pick a random edge to flip
        e = random.randint(0, n_edges - 1)
        
        # Calculate delta
        delta_s = 1 if not selected[e] else -1
        
        # Check violations change
        delta_v = 0
        affected_faces = edge_faces[e]
        
        # Store old counts to revert if needed? 
        # We can calculate delta_v directly.
        
        new_violations = current_violations
        
        # If adding edge e:
        if not selected[e]:
            # Check if any affected face becomes violated (count 3 -> 4)
            for f in affected_faces:
                if face_count[f] == 3:
                    # This face will become violated
                    # But wait, if it was already violated, it stays violated.
                    # If it was not violated (count < 3), it becomes violated?
                    # face_count tracks the number of selected edges in the face.
                    # Violation is face_count == 4.
                    
                    # If face_count[f] == 3, adding e makes it 4.
                    # Was it violated before? No, because 3 != 4.
                    # So new violation.
                    new_violations += 1
                # If face_count[f] < 3, no violation.
                # If face_count[f] == 4, it was already violated. Adding e makes it 5? 
                # No, we only add if not selected.
                # But can a face have count > 4? No, max 4 edges.
                # So if count was 4, it's violated. Adding e is impossible if e is in face?
                # Wait, e is in affected_faces. So e is one of the 4 edges.
                # If face_count[f] == 4, then e must be selected?
                # No, face_count is sum of selected edges in face.
                # If e is not selected, and face_count[f] == 4, then the other 3 are selected.
                # Adding e makes count 5? No, face has only 4 edges.
                # So if e is not selected, max count is 3?
                # No, if e is not selected, the other 3 CAN be selected. Count = 3.
                # So face_count[f] cannot be 4 if e is not selected.
                # Therefore, if not selected[e], face_count[f] <= 3 for all f in edge_faces[e].
                
        # If removing edge e:
        else:
            for f in affected_faces:
                if face_count[f] == 4:
                    # It was violated. Removing e makes count 3. No longer violated.
                    new_violations -= 1
                # If face_count[f] < 4, it was not violated. Removing e makes it < 4. Still not violated.
        
        delta_v = new_violations - current_violations
        
        new_score = current_state_score + delta_s - PENALTY * delta_v
        
        # Acceptance criterion
        delta_score = new_score - current_state_score
        
        if delta_score >= 0:
            accept = True
        else:
            accept = random.random() < math.exp(delta_score / T)
            
        if accept:
            # Apply move
            selected[e] = not selected[e]
            for f in affected_faces:
                face_count[f] += delta_s
            current_violations = new_violations
            current_state_score = new_score
            
            # Update best
            if current_violations == 0:
                if current_state_score > best_score:
                    best_score = current_state_score
                    best_selected = selected.copy()
                    best_face_count = face_count.copy()
        
        # Occasional restart if stuck?
        # Or just let SA handle it.
        
    # Build result from best valid solution
    # Ensure best_selected is valid (violations == 0)
    # If not, we might have a bug. But we only update best if violations == 0.
    # If best_score is from a state with violations > 0, it's invalid.
    # We initialized best_score with a valid state.
    
    result = []
    for i, (u, v) in enumerate(edges):
        if best_selected[i]:
            result.append([u, v])
    
    return result
