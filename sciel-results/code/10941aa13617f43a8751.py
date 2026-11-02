"""Rewritten program for packing 12 unit hexagons in a regular hexagon with fast vectorized objective."""
import numpy as np
import time
import random

N = 12
SQRT3 = np.sqrt(3.0)
TWO_SQRT3 = 2.0 * SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons.
        inner: list of (x, y, angle_deg)
        center: (cx, cy)
        angle: rotation angle of outer hexagon in degrees
        Returns the minimum side length of the outer hexagon.
        """
        # Precompute all 72 vertices at once
        cx, cy = center
        all_verts = np.empty((N * 6, 2))
        for i in range(N):
            px, py, pa = inner[i]
            pa_rad = np.radians(pa)
            # 6 vertices of unit hexagon centered at (px, py) with rotation pa
            for k in range(6):
                theta = pa_rad + k * np.pi / 3.0
                all_verts[i*6 + k, 0] = px + np.cos(theta)
                all_verts[i*6 + k, 1] = py + np.sin(theta)
        
        # Subtract center
        all_verts[:, 0] -= cx
        all_verts[:, 1] -= cy
        
        # 6 normals of the outer hexagon
        angle_rad = np.radians(angle)
        # For a regular hexagon, the support function in direction n gives the extent
        # The side length s relates to the support: s = max_extent * 2 / sqrt(3)
        # The 6 outward normals of a regular hexagon (pointy-top or flat-top depends on angle)
        # Normal k points at angle: angle + 30 + k*60 degrees (for flat-top at angle=0)
        normals = np.empty((6, 2))
        for k in range(6):
            na = angle_rad + np.pi/6.0 + k * np.pi / 3.0
            normals[k, 0] = np.cos(na)
            normals[k, 1] = np.sin(na)
        
        # For each normal, find max dot product (extent)
        max_extent = 0.0
        for k in range(6):
            n = normals[k]
            dots = all_verts[:, 0] * n[0] + all_verts[:, 1] * n[1]
            m = np.max(dots)
            if m > max_extent:
                max_extent = m
        
        return max_extent * TWO_SQRT3

    def disjoint_penalty_fast(inner):
        """Compute a smooth penalty for overlapping hexagons using SAT.
        Returns 0 if all disjoint, positive value if overlaps exist.
        """
        # Precompute vertices
        verts = np.empty((N, 6, 2))
        for i in range(N):
            px, py, pa = inner[i]
            pa_rad = np.radians(pa)
            for k in range(6):
                theta = pa_rad + k * np.pi / 3.0
                verts[i, k, 0] = px + np.cos(theta)
                verts[i, k, 1] = py + np.sin(theta)
        
        penalty = 0.0
        for i in range(N):
            vi = verts[i]
            # Compute edge normals for hexagon i
            ni = np.empty((6, 2))
            for k in range(6):
                e = vi[(k+1) % 6] - vi[k]
                ni[k, 0] = -e[1]
                ni[k, 1] = e[0]
                nn = np.sqrt(ni[k,0]*ni[k,0] + ni[k,1]*ni[k,1])
                if nn > 1e-12:
                    ni[k, 0] /= nn
                    ni[k, 1] /= nn
            
            for j in range(i+1, N):
                vj = verts[j]
                
                # Quick reject: center distance
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                if dx*dx + dy*dy > 4.0:
                    continue
                
                # SAT check using normals from i
                separated = False
                for k in range(6):
                    n = ni[k]
                    proj_i = np.max(vi[:, 0] * n[0] + vi[:, 1] * n[1])
                    proj_j_min = np.min(vj[:, 0] * n[0] + vj[:, 1] * n[1])
                    if proj_i < proj_j_min - 1e-9:
                        separated = True
                        break
                    proj_j_max = np.max(vj[:, 0] * n[0] + vj[:, 1] * n[1])
                    proj_i_min = np.min(vi[:, 0] * n[0] + vi[:, 1] * n[1])
                    if proj_j_max < proj_i_min - 1e-9:
                        separated = True
                        break
                
                if separated:
                    continue
                
                # Check normals from j
                nj = np.empty((6, 2))
                for k in range(6):
                    e = vj[(k+1) % 6] - vj[k]
                    nj[k, 0] = -e[1]
                    nj[k, 1] = e[0]
                    nn = np.sqrt(nj[k,0]*nj[k,0] + nj[k,1]*nj[k,1])
                    if nn > 1e-12:
                        nj[k, 0] /= nn
                        nj[k, 1] /= nn
                
                for k in range(6):
                    n = nj[k]
                    proj_i_max = np.max(vi[:, 0] * n[0] + vi[:, 1] * n[1])
                    proj_j_min = np.min(vj[:, 0] * n[0] + vj[:, 1] * n[1])
                    if proj_i_max < proj_j_min - 1e-9:
                        separated = True
                        break
                    proj_j_max = np.max(vj[:, 0] * n[0] + vj[:, 1] * n[1])
                    proj_i_min = np.min(vi[:, 0] * n[0] + vi[:, 1] * n[1])
                    if proj_j_max < proj_i_min - 1e-9:
                        separated = True
                        break
                
                if not separated:
                    # Overlap detected - add penalty
                    penalty += 1.0
        
        return penalty

    def objective(params):
        """Compute objective: outer side length + penalty for overlaps."""
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        pen = disjoint_penalty_fast(inner)
        if pen > 0:
            return 100.0 + pen * 10.0
        
        side = compute_min_side_fast(inner, center, angle)
        return side

    def initial_guess():
        """Generate initial hexagonal packing configuration."""
        params = np.zeros(3*N + 3)
        positions = []
        # 1 center, 6 in ring 1 (distance 2), 5 in ring 2
        positions.append((0.0, 0.0, 0.0))
        for i in range(6):
            theta = i * np.pi / 3.0
            positions.append((2.0*np.cos(theta), 2.0*np.sin(theta), 0.0))
        for i in range(5):
            theta = i * 2.0*np.pi / 5.0 + np.pi/5.0
            positions.append((4.0*np.cos(theta), 4.0*np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    # Multiple restarts with different initial configurations
    def generate_restart_config(restart_id):
        """Generate varied initial configurations."""
        params = np.zeros(3*N + 3)
        rng = random.Random(restart_id * 12345 + 67890)
        
        if restart_id == 0:
            # Standard hexagonal packing
            positions = [(0.0, 0.0, 0.0)]
            for i in range(6):
                theta = i * np.pi / 3.0
                positions.append((2.0*np.cos(theta), 2.0*np.sin(theta), 0.0))
            for i in range(5):
                theta = i * 2.0*np.pi / 5.0 + np.pi/5.0
                positions.append((4.0*np.cos(theta), 4.0*np.sin(theta), 0.0))
        elif restart_id == 1:
            # Slightly rotated configuration
            positions = [(0.0, 0.0, 30.0)]
            for i in range(6):
                theta = i * np.pi / 3.0 + 0.1
                positions.append((2.0*np.cos(theta), 2.0*np.sin(theta), 30.0))
            for i in range(5):
                theta = i * 2.0*np.pi / 5.0 + np.pi/5.0 + 0.2
                positions.append((4.0*np.cos(theta), 4.0*np.sin(theta), 30.0))
        elif restart_id == 2:
            # Linear-ish arrangement
            for i in range(6):
                positions.append((i*2.0, 0.0, 0.0))
            for i in range(6):
                positions.append((i*2.0, 2.0*SQRT3, 0.0))
        elif restart_id == 3:
            # Random perturbation of hexagonal
            positions = [(0.0, 0.0, 0.0)]
            for i in range(6):
                theta = i * np.pi / 3.0
                r = 2.0 + rng.gauss(0, 0.1)
                positions.append((r*np.cos(theta), r*np.sin(theta), 0.0))
            for i in range(5):
                theta = i * 2.0*np.pi / 5.0 + np.pi/5.0
                r = 4.0 + rng.gauss(0, 0.2)
                positions.append((r*np.cos(theta), r*np.sin(theta), 0.0))
        else:
            # More random
            base = [(0.0, 0.0, 0.0)]
            for i in range(6):
                theta = i * np.pi / 3.0
                base.append((2.0*np.cos(theta), 2.0*np.sin(theta), 0.0))
            for i in range(5):
                theta = i * 2.0*np.pi / 5.0 + np.pi/5.0
                base.append((4.0*np.cos(theta), 4.0*np.sin(theta), 0.0))
            positions = []
            for (px, py, pa) in base:
                positions.append((px + rng.gauss(0, 0.5), py + rng.gauss(0, 0.5), pa + rng.gauss(0, 10)))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def simulated_annealing(params, time_limit, start_temp=1.0):
        """Run simulated annealing optimization."""
        current = params.copy()
        current_score = objective(current)
        best = current.copy()
        best_score = current_score
        
        T = start_temp
        T_min = 0.0005
        cooling = 0.9998
        step_base = 0.4
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            new = current.copy()
            n_perturb = 1 if random.random() < 0.6 else (2 if random.random() < 0.8 else 3)
            for _ in range(n_perturb):
                idx = random.randint(0, 3*N + 2)
                s = step_base * (0.3 + 0.7 * T)
                new[idx] += random.gauss(0, s)
            
            new_score = objective(new)
            
            if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
                current = new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = current.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best, best_score

    # Main optimization loop with multiple restarts
    best_overall = None
    best_overall_score = 1e10
    
    n_restarts = 8
    time_per_restart = (deadline - start_time - 15.0) / n_restarts
    
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        params = generate_restart_config(r)
        score = objective(params)
        
        if score < best_overall_score:
            best_overall = params.copy()
            best_overall_score = score
        
        # Run SA for this restart
        remaining = deadline - time.time() - 15.0
        if remaining > 5.0:
            t_alloc = min(time_per_restart, remaining * 0.8)
            result, result_score = simulated_annealing(params, t_alloc, start_temp=0.5 + 0.3*r)
            
            if result_score < best_overall_score:
                best_overall = result.copy()
                best_overall_score = result_score
    
    # Final polish on best solution
    if time.time() < deadline - 5.0 and best_overall is not None:
        polish_time = min(10.0, deadline - time.time() - 3.0)
        if polish_time > 1.0:
            # Fine-tune with very small steps
            current = best_overall.copy()
            current_score = best_overall_score
            T = 0.01
            end_time = time.time() + polish_time
            while time.time() < end_time:
                new = current.copy()
                idx = random.randint(0, 3*N + 2)
                new[idx] += random.gauss(0, 0.02)
                new_score = objective(new)
                if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
                    current = new
                    current_score = new_score
                    if current_score < best_overall_score:
                        best_overall_score = current_score
                        best_overall = current.copy()
                T *= 0.999
                if T < 0.001:
                    T = 0.001
    
    # Extract final answer
    if best_overall is None:
        best_overall = initial_guess()
    
    inner = []
    for i in range(N):
        inner.append([float(best_overall[3*i]), float(best_overall[3*i+1]), float(best_overall[3*i+2])])
    
    center = (float(best_overall[3*N]), float(best_overall[3*N+1]))
    angle = float(best_overall[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Validate: check disjointness
    valid = True
    for i in range(N):
        vi = inner[i]
        for j in range(i+1, N):
            vj = inner[j]
            dx = vi[0] - vj[0]
            dy = vi[1] - vj[1]
            if dx*dx + dy*dy <= 4.0:
                # Need detailed check
                vi_verts = np.array([[vi[0] + np.cos(np.radians(vi[2]) + k*np.pi/3.0),
                                       vi[1] + np.sin(np.radians(vi[2]) + k*np.pi/3.0)] for k in range(6)])
                vj_verts = np.array([[vj[0] + np.cos(np.radians(vj[2]) + k*np.pi/3.0),
                                       vj[1] + np.sin(np.radians(vj[2]) + k*np.pi/3.0)] for k in range(6)])
                separated = False
                for k in range(6):
                    e1 = vi_verts[(k+1)%6] - vi_verts[k]
                    n1 = np.array([-e1[1], e1[0]])
                    nn1 = np.linalg.norm(n1)
                    if nn1 > 1e-12:
                        n1 = n1 / nn1
                        p1 = np.max(np.dot(vi_verts, n1))
                        p2 = np.min(np.dot(vj_verts, n1))
                        if p1 < p2 - 1e-9 or np.max(np.dot(vj_verts, n1)) < np.min(np.dot(vi_verts, n1)) - 1e-9:
                            separated = True
                            break
                    e2 = vj_verts[(k+1)%6] - vj_verts[k]
                    n2 = np.array([-e2[1], e2[0]])
                    nn2 = np.linalg.norm(n2)
                    if nn2 > 1e-12:
                        n2 = n2 / nn2
                        p1 = np.max(np.dot(vi_verts, n2))
                        p2 = np.min(np.dot(vj_verts, n2))
                        if p1 < p2 - 1e-9 or np.max(np.dot(vj_verts, n2)) < np.min(np.dot(vi_verts, n2)) - 1e-9:
                            separated = True
                            break
                if not separated:
                    valid = False
                    break
        if not valid:
            break
    
    if not valid:
        # Fallback to known valid configuration
        inner = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = i * np.pi / 3.0
            inner.append([2.0*np.cos(theta), 2.0*np.sin(theta), 0.0])
        for i in range(5):
            theta = i * 2.0*np.pi / 5.0 + np.pi/5.0
            inner.append([4.0*np.cos(theta), 4.0*np.sin(theta), 0.0])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)
    
    return inner, center, side, angle
