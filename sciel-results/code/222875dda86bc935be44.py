<<<<<<< SEARCH
            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                fvals[i] = func(pop[i])
                calls_used += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls_used >= budget:
                break

            # Sort
            order = np.argsort(fvals)
            y = np.zeros((mu, dim))
            for i in range(mu):
                y[i] = pop[order[i]]
=======
            # Evaluate population
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= budget:
                    break
                fvals[i] = func(pop[i])
                calls_used += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()

            if calls_used >= budget:
                break

            # Sort
            order = np.argsort(fvals)
            
            # Stall detection: if no improvement in recent generations, boost sigma
            if not hasattr(self, '_gen'):
                self._gen = 0
                self._last_improve_gen = 0
                self._stall_count = 0
            self._gen += 1
            if fvals[order[0]] < best_f:
                self._last_improve_gen = self._gen
                self._stall_count = 0
            elif self._gen - self._last_improve_gen > 5:
                sigma *= 1.5
                self._stall_count += 1
                if self._stall_count > 3:
                    # Reset to best found with fresh parameters
                    mean = best_x.copy()
                    sigma = 2.0
                    C = np.eye(dim)
                    ps = np.zeros(dim)
                    pc = np.zeros(dim)
                    self._stall_count = 0

            y = np.zeros((mu, dim))
            for i in range(mu):
                y[i] = pop[order[i]]
>>>>>>> REPLACE
