<<<<<<< SEARCH
    # Phase 3: Simulated annealing for some candidates to escape local minima
        if time.time() < deadline - 20.0 and idx % 3 == 0:
            sa_pts, sa_energy = _simulated_annealing(newton_pts, deadline - 15.0, rng, max_iter=200)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
            # Polish after SA
            if time.time() < deadline - 10.0:
                polished, pol_e = _newton_polish(sa_pts, max_iter=100, deadline=deadline - 5.0)
                if pol_e < best_energy:
                    best_energy = pol_e
                    best_pts = polished.copy()
=======
    # Phase 3: Simulated annealing for some candidates to escape local minima
        if time.time() < deadline - 20.0 and idx % 2 == 0:
            sa_pts, sa_energy = _simulated_annealing(newton_pts, deadline - 15.0, rng, max_iter=300)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
            # Polish after SA
            if time.time() < deadline - 10.0:
                polished, pol_e = _newton_polish(sa_pts, max_iter=150, deadline=deadline - 5.0)
                if pol_e < best_energy:
                    best_energy = pol_e
                    best_pts = polished.copy()
>>>>>>> REPLACE

