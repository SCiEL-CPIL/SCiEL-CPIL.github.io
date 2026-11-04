<<<<<<< SEARCH
    # Simulated Annealing phase
    T = 0.8
    T_end = 0.003
    alpha_sa = 0.9997
    
    # Time allocation: SA gets most time, then LBFGS polish
    # With faster SAT (center-distance rejection), we can afford more SA iterations
    sa_time_budget = TIME_LIMIT - 20.0  # leave 20s for LBFGS + safety
    
    while time.time() - t0 < sa_time_budget:
        r = random.random()
        
        if r < 0.45:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 4.0 + 3.0 * T)
        elif r < 0.75:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.08 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 1.5 + 1.5 * T)
        else:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.12 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                if r < 0.45:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.65:
                    cfg[idx][2] = old_ang
                elif r < 0.75:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.85:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        
        T = max(T_end, T * alpha_sa)
=======
    # Simulated Annealing phase with soft penalties
    T = 0.8
    T_end = 0.002
    alpha_sa = 0.9996
    
    # Time allocation: SA gets most time, then LBFGS polish
    sa_time_budget = TIME_LIMIT - 25.0  # leave 25s for LBFGS + outer rotation + safety
    
    def soft_score(cfg_local, center_local, angle_local):
        """Return (score, penalty) where score is outer side and penalty is overlap measure."""
        # Compute outer side
        outer = _outer_hex_side(cfg_local, tuple(center_local), angle_local)
        # Compute overlap penalty
        total_pen = 0.0
        m = len(cfg_local)
        for i in range(m):
            for j in range(i + 1, m):
                d = _sdf_distance(cfg_local[i][0], cfg_local[i][1], cfg_local[i][2], 
                                  cfg_local[j][0], cfg_local[j][1], cfg_local[j][2])
                if d < 0:
                    total_pen += max(0, -d)
        return outer, total_pen
    
    while time.time() - t0 < sa_time_budget:
        r = random.random()
        
        if r < 0.40:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.12 * (1.0 + 0.3 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.58:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 3.0 + 2.0 * T)
        elif r < 0.70:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.06 * (1.0 + 0.3 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.82:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 1.0 + 1.0 * T)
        else:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.10 * (1.0 + 0.3 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        
        # Use soft score for SA acceptance
        new_outer, new_pen = soft_score(cfg, tuple(center), angle)
        new_soft = new_outer + 50.0 * new_pen
        
        if new_soft < cur_score or (T > T_end and random.random() < math.exp(-(new_soft - cur_score) / T)):
            cur_score = new_soft
            # Check if actually valid and update best
            actual_score = full_score(cfg, tuple(center), angle)
            if actual_score is not None and actual_score < best_score:
                best_score = actual_score
                best_cfg = [row[:] for row in cfg]
                best_center = center[:]
                best_angle = angle
        else:
            if r < 0.40:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.58:
                cfg[idx][2] = old_ang
            elif r < 0.70:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.82:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        
        T = max(T_end, T * alpha_sa)
>>>>>>> REPLACE

