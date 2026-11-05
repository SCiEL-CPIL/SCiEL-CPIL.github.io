<<<<<<< SEARCH
            # Pick an unsatisfied clause (weighted by fewest unsat literals for better heuristics)
            # Sample a few random unsat clauses and pick the one with min unsat literals
            sample_size = min(3, len(unsat_set))
            if sample_size <= 1:
                c_idx = next(iter(unsat_set))
            else:
                candidates = list(rng.sample(unsat_set, sample_size))
=======
            # Pick an unsatisfied clause (weighted by fewest unsat literals for better heuristics)
            # Sample a few random unsat clauses and pick the one with min unsat literals
            sample_size = min(3, len(unsat_set))
            if sample_size <= 1:
                c_idx = next(iter(unsat_set))
            else:
                candidates = list(rng.sample(list(unsat_set), sample_size))
>>>>>>> REPLACE

