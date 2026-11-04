<<<<<<< SEARCH
        result = differential_evolution(
            objective,
            bounds,
            max_iter=max_iter,
            popsize=15,
            tol=0.01,
            seed=42 + cfg_idx,
            polish=False,
            init='sobol',
        )
=======
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=15,
            tol=0.01,
            seed=42 + cfg_idx,
            polish=False,
            init='sobol',
        )
>>>>>>> REPLACE
BET: {mechanism: "Fixing the typo max_iter to maxiter allows differential_evolution to run, enabling the search to actually execute and find a valid solution instead of crashing immediately", q: 0.95, delta: [1.0, 1.2]}
