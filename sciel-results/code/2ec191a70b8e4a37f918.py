import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        
        N = 20
        if self.budget < N:
            N = self.budget
            
        rng = np.random.default_rng(0)
        P = lo + (hi - lo) * rng.random((N, n))
        F = np.array([problem(x) for x in P])
        evals = N
        
        eta_c = 20.0
        eta_m = 20.0
        pc = 0.9
        
        while evals < self.budget:
            # Pre-compute ranks and crowding distances for P
            fronts = self._fast_non_dominated_sort(F)
            rank = np.zeros(N, dtype=int)
            for r, front in enumerate(fronts):
                rank[front] = r
            
            # Compute crowding distance for each front
            crowd_dist = np.zeros(N)
            for front in fronts:
                if len(front) > 2:
                    front_F = F[front]
                    d = self._crowding_distance(front_F)
                    crowd_dist[front] = d
                elif len(front) == 2:
                    crowd_dist[front] = np.inf
                else:
                    crowd_dist[front] = np.inf

            # Create offspring Q
            Q = np.empty_like(P)
            Fq = np.empty_like(F)
            q_idx = 0
            while q_idx < N and evals < self.budget:
                p1_idx = self._crowded_tournament(rank, crowd_dist, rng)
                p2_idx = self._crowded_tournament(rank, crowd_dist, rng)
                p1 = P[p1_idx]
                p2 = P[p2_idx]
                
                if rng.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()
                
                c1 = self._polynomial_mutation(c1, lo, hi, eta_m, rng)
                c2 = self._polynomial_mutation(c2, lo, hi, eta_m, rng)
                
                Q[q_idx] = c1
                Fq[q_idx] = problem(c1)
                evals += 1
                q_idx += 1
                if q_idx < N and evals < self.budget:
                    Q[q_idx] = c2
                    Fq[q_idx] = problem(c2)
                    evals += 1
                    q_idx += 1
                elif q_idx < N:
                    break
            
            if evals >= self.budget:
                break
                
            # Combine P and Q
            R = np.vstack([P, Q])
            Fr = np.vstack([F, Fq])
            
            # Non-dominated sorting on R
            fronts_R = self._fast_non_dominated_sort(Fr)
            
            # Select next population P
            P_new = np.empty_like(P)
            F_new = np.empty_like(F)
            p_idx = 0
            for front in fronts_R:
                if p_idx + len(front) <= N:
                    P_new[p_idx:p_idx+len(front)] = R[front]
                    F_new[p_idx:p_idx+len(front)] = Fr[front]
                    p_idx += len(front)
                else:
                    front_F = Fr[front]
                    distances = self._crowding_distance(front_F)
                    sorted_indices = np.argsort(-distances)
                    needed = N - p_idx
                    selected_front_indices = front[sorted_indices[:needed]]
                    P_new[p_idx:p_idx+needed] = R[selected_front_indices]
                    F_new[p_idx:p_idx+needed] = Fr[selected_front_indices]
                    p_idx += needed
                    break
            
            P = P_new
            F = F_new

        return P

    def _crowded_tournament(self, rank, crowd_dist, rng):
        i = rng.integers(len(rank))
        j = rng.integers(len(rank))
        if rank[i] < rank[j]:
            return i
        elif rank[j] < rank[i]:
            return j
        else:
            if crowd_dist[i] > crowd_dist[j]:
                return i
            elif crowd_dist[j] > crowd_dist[i]:
                return j
            else:
                return i if rng.random() < 0.5 else j

    def _sbx_crossover(self, x1, x2, lo, hi, eta, rng):
        c1 = np.empty_like(x1)
        c2 = np.empty_like(x2)
        for i in range(len(x1)):
            if rng.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = ((alpha + 1.0) ** (eta + 1.0))
                        ratio = (1.0 + (x2[i] - x1[i]) / (hi[i] - lo[i])) ** beta
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = ((alpha + 1.0) ** (eta + 1.0))
                        ratio = (1.0 - (x1[i] - x2[i]) / (hi[i] - lo[i])) ** beta
                    c1[i] = 0.5 * ((1.0 + ratio) * x1[i] + (1.0 - ratio) * x2[i])
                    c2[i] = 0.5 * ((1.0 - ratio) * x1[i] + (1.0 + ratio) * x2[i])
                else:
                    c1[i] = x1[i]
                    c2[i] = x2[i]
            else:
                c1[i] = x2[i]
                c2[i] = x1[i]
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, eta, rng):
        x_mut = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.5:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                u = rng.random()
                if u < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** (1.0 / (eta + 1.0))
                x_mut[i] = x[i] + deltaq * (hi[i] - lo[i])
        return np.clip(x_mut, lo, hi)

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominated = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominated[i] += 1
        
        front = []
        for i in range(n):
            if n_dominated[i] == 0:
                front.append(i)
        
        while front:
            fronts.append(front)
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            front = next_front
        return fronts

    def _crowding_distance(self, F):
        n = len(F)
        if n == 0: return np.array([])
        if n == 1: return np.array([np.inf])
        if n == 2: return np.array([np.inf, np.inf])
        
        m = F.shape[1]
        dist = np.zeros(n)
        for k in range(m):
            sorted_idx = np.argsort(F[:, k])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_min = F[sorted_idx[0], k]
            f_max = F[sorted_idx[-1], k]
            if f_max == f_min: continue
            for i in range(1, n-1):
                dist[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / (f_max - f_min)
        return dist
