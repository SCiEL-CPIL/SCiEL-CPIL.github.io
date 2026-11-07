"""Multi-objective optimizer: LHS initialization + fast HV-guided DE with local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Helper to check dominance
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        # Helper to get non-dominated set indices
        def get_nondominated_idx(F):
            n = len(F)
            nd_idx = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i != j and dominates(F[j], F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nd_idx.append(i)
            return nd_idx

        # Fast approximate HV contribution: use 3D sweep-line HV, O(n^2 log n)
        def calc_hv_3d(points, ref):
            """Exact 3D hypervolume via sweep line. points: list of [x,y,z], ref: [rx,ry,rz]"""
            if len(points) == 0:
                return 0.0
            pts = sorted(points, key=lambda p: p[0])
            xs = sorted(set([p[0] for p in pts] + [ref[0]]))
            total = 0.0
            for k in range(len(xs) - 1):
                x1, x2 = xs[k], xs[k+1]
                dx = x2 - x1
                if dx <= 0:
                    continue
                active = [p for p in pts if p[0] <= x1]
                if not active:
                    continue
                # 2D union area of [p[1],ry] x [p[2],rz]
                ys = sorted(set([p[1] for p in active] + [ref[1]]))
                area = 0.0
                for m in range(len(ys) - 1):
                    y1, y2 = ys[m], ys[m+1]
                    dy = y2 - y1
                    if dy <= 0:
                        continue
                    act2 = [p for p in active if p[1] <= y1]
                    if not act2:
                        continue
                    intervals = sorted([(p[2], ref[2]) for p in act2])
                    merged = [list(intervals[0])]
                    for iv in intervals[1:]:
                        if iv[0] <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], iv[1])
                        else:
                            merged.append(list(iv))
                    length = sum(m2[1] - m2[0] for m2 in merged)
                    area += dy * length
                total += dx * area
            return total

        def hv_contribs_fast(F_nd, ref):
            """Marginal HV contribution of each point. O(n^3) worst case but fast for n<=30."""
            n = len(F_nd)
            if n == 0:
                return np.array([])
            pts = [list(f) for f in F_nd]
            hv_full = calc_hv_3d(pts, ref)
            contribs = np.zeros(n)
            for i in range(n):
                S_i = pts[:i] + pts[i+1:]
                hv_i = calc_hv_3d(S_i, ref)
                contribs[i] = hv_full - hv_i
            return contribs

        # Phase 1: Latin Hypercube Sampling for initial coverage
        used = 0
        all_X = []
        all_F = []
        
        # LHS: n_init points, each dimension stratified
        n_lhs = 64  # 4^3, good balance
        if n_lhs > budget:
            n_lhs = min(budget, 49)  # 7^2
        
        # Generate LHS points
        np.random.seed(np.random.randint(10000))  # deterministic sub-seed for LHS
        lhs_points = []
        for i in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling: divide [lo,hi] into n_lhs bins, pick one per bin
                bin_idx = (i + d) % n_lhs  # simple Latin hypercube
                x[d] = lo[d] + (hi[d] - lo[d]) * (bin_idx + np.random.random()) / n_lhs
            lhs_points.append(x)
        
        # Evaluate LHS points
        for x in lhs_points:
            if used >= budget or time.time() > deadline:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
        
        # Get initial non-dominated archive
        nd_idx = get_nondominated_idx(all_F)
        archive_X = [all_X[i] for i in nd_idx]
        archive_F = [all_F[i] for i in nd_idx]
        
        # Phase 2: DE with HV-guided parent selection
        CR = 0.8
        F_base = 0.5
        
        while used < budget and time.time() < deadline:
            if len(archive_F) < 2:
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1
                is_dom = any(dominates(af, f) for af in archive_F)
                if not is_dom:
                    dom_idx = [i for i, af in enumerate(archive_F) if dominates(f, af)]
                    for i in sorted(dom_idx, reverse=True):
                        archive_X.pop(i)
                        archive_F.pop(i)
                    archive_X.append(x)
                    archive_F.append(f)
                continue
            
            # Keep archive manageable: if too large, prune by HV contribution
            if len(archive_F) > 25:
                contribs = hv_contribs_fast(archive_F, ref_point)
                top_idx = np.argsort(contribs)[-20:]
                archive_X = [archive_X[i] for i in top_idx]
                archive_F = [archive_F[i] for i in top_idx]
            
            # Compute contributions for parent selection
            n_arch = len(archive_F)
            if n_arch > 15:
                contribs = hv_contribs_fast(archive_F, ref_point)
            else:
                contribs = hv_contribs_fast(archive_F, ref_point)
            
            # Select parents: best contributor + 2 random
            idx1 = np.argmax(contribs)
            if n_arch >= 3:
                remaining = list(range(n_arch))
                remaining.remove(idx1)
                idx2 = remaining[np.random.randint(len(remaining))]
                remaining.remove(idx2)
                idx3 = remaining[np.random.randint(len(remaining))]
            else:
                idx2 = np.random.randint(n_arch)
                idx3 = np.random.randint(n_arch)
                while idx3 == idx2:
                    idx3 = np.random.randint(n_arch)
            
            p1 = np.array(archive_X[idx1])
            p2 = np.array(archive_X[idx2])
            p3 = np.array(archive_X[idx3])
            
            # Mutation with adaptive F
            F_mut = F_base + 0.15 * np.random.random()
            v = p1 + F_mut * (p2 - p3)
            
            # Crossover
            j_rand = np.random.randint(dim)
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = p1[j]
            
            trial = np.clip(trial, lo, hi)
            
            if time.time() > deadline:
                break
            f_trial = problem(trial)
            all_X.append(trial)
            all_F.append(f_trial)
            used += 1
            
            # Update archive
            is_dom = any(dominates(af, f_trial) for af in archive_F)
            if not is_dom:
                dom_idx = [i for i, af in enumerate(archive_F) if dominates(f_trial, af)]
                for i in sorted(dom_idx, reverse=True):
                    archive_X.pop(i)
                    archive_F.pop(i)
                archive_X.append(trial)
                archive_F.append(f_trial)
        
        # Phase 3: Local polish around best HV contributors
        if used < budget and time.time() < deadline:
            if len(archive_F) >= 2:
                contribs = hv_contribs_fast(archive_F, ref_point)
                top_k = min(5, len(archive_F))
                top_idx = np.argsort(contribs)[-top_k:]
                
                for idx in top_idx:
                    if used >= budget or time.time() > deadline:
                        break
                    base_x = np.array(archive_X[idx])
                    # Local random perturbation
                    n_local = min(10, budget - used)
                    for _ in range(n_local):
                        if used >= budget or time.time() > deadline:
                            break
                        step_size = 0.05 * np.random.random()
                        pert = (np.random.random(dim) - 0.5) * 2 * step_size
                        x = np.clip(base_x + pert, lo, hi)
                        f = problem(x)
                        all_X.append(x)
                        all_F.append(f)
                        used += 1
                        is_dom = any(dominates(af, f) for af in archive_F)
                        if not is_dom:
                            dom_idx2 = [i for i, af in enumerate(archive_F) if dominates(f, af)]
                            for i in sorted(dom_idx2, reverse=True):
                                archive_X.pop(i)
                                archive_F.pop(i)
                            archive_X.append(x)
                            archive_F.append(f)
        
        return all_X
# EVOLVE-BLOCK-END
