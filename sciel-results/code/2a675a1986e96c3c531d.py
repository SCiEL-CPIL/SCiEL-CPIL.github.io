    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        pop_size = min(20, budget)
        if pop_size == 0:
            return np.array([])

        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Archive of non-dominated points
        archive = []

        def is_dominated(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def add_to_archive(x, f):
            for _, fa in archive:
                if is_dominated(fa, f):
                    return
            i = 0
            while i < len(archive):
                _, fa = archive[i]
                if is_dominated(f, fa):
                    archive.pop(i)
                else:
                    i += 1
            archive.append((x, f))

        for x, f in zip(X, F):
            add_to_archive(x, f)

        while used < budget:
            if time.time() > deadline:
                break

            # DE/rand/1
            for i in range(pop_size):
                if used >= budget:
                    break
                
                # Select 3 distinct random indices
                indices = [i]
                while len(indices) < 3:
                    r = np.random.randint(pop_size)
                    if r not in indices:
                        indices.append(r)
                r1, r2, r3 = indices
                
                v = X[r1] + 0.5 * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)
                
                # Mutation
                if np.random.random() < 0.1:
                    v += 0.1 * (hi - lo) * np.random.standard_normal(dim)
                    v = np.clip(v, lo, hi)
                    
                fv = problem(v)
                used += 1
                add_to_archive(v, fv)
                
                # Replace parent if offspring is not dominated by parent
                if not is_dominated(F[i], fv):
                    X[i] = v
                    F[i] = fv

        return X
