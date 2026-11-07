    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        d = self.dim
        self.func = func
        self.calls = 0
        
        best_x = None
        best_f = float('inf')
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(d))
        lam = max(4, min(lam, 20))
        
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        c1 = 2 / ((d + 1.3)**2 + lam)
        cmu = min(1 - c1, 2 * (mueff - 2) / ((d + 2)**2 + lam))
        cc = (4 + lam/d) / (d + 4 + 2*lam/d)
        cs = (lam/2 + 1) / (d + lam/2 + 5)
        dsigma = 1 + 2*max(0, np.sqrt(mueff/d - 1)) + cs
        
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        def sample_point():
            z = np.random.normal(0, 1, d)
            L = np.linalg.cholesky(C)
            x = m + sigma * L @ z
            return x
        
        while self.calls < self.budget:
            if time.time() > deadline - 10:
                break
                
            pop = []
            pop_f = []
            
            for i in range(lam):
                if self.calls >= self.budget:
                    break
                x = sample_point()
                x = np.clip(x, -5, 5)
                f = self._evaluate(x)
                if f is None:
                    break
                pop.append(x)
                pop_f.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(pop) < 2:
                break
                
            indices = np.argsort(pop_f)
            pop = [pop[i] for i in indices]
            pop_f = [pop_f[i] for i in indices]
            
            m_old = m.copy()
            for i in range(mu):
                m += weights[i] * (pop[i] - m_old)
            
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C_inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals)) @ eigvecs.T
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (m - m_old) / sigma
            
            norm_ps = np.linalg.norm(ps)
            denom = np.sqrt(1 - (1-cs)**2)
            hsig = 1 if norm_ps / denom < (1.4 + 2/(d+1)) * denom else 0
            
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (m - m_old) / sigma
            
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                diff = pop[i] - m_old
                rank_mu += weights[i] * np.outer(diff, diff)
            
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            C = (C + C.T) / 2
            
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.any(eigvals < 1e-16):
                C = C + 1e-16 * np.eye(d)
            
            sigma *= np.exp(cs/dsigma * (np.linalg.norm(ps) / np.sqrt(2*d) - 1))
            
            if sigma < 1e-8:
                break

        if best_x is not None and self.calls < self.budget and time.time() < deadline - 5:
            x_polish = best_x.copy()
            f_polish = best_f
            step_size = 0.1
            
            for _ in range(10):
                if self.calls >= self.budget or time.time() > deadline - 2:
                    break
                improved = False
                for i in range(d):
                    if self.calls >= self.budget:
                        break
                    x_try = x_polish.copy()
                    x_try[i] += step_size
                    x_try = np.clip(x_try, -5, 5)
                    f_try = self._evaluate(x_try)
                    if f_try is not None and f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved = True
                        continue
                    x_try = x_polish.copy()
                    x_try[i] -= step_size
                    x_try = np.clip(x_try, -5, 5)
                    f_try = self._evaluate(x_try)
                    if f_try is not None and f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved = True
                
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break
                
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_f = self.func(best_x)
            
        return best_x, best_f
