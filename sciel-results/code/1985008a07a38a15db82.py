<<<<<<< SEARCH
    # Multi-restart: each restart gets greedy init + SA
    time_per_restart = sa_total_time / num_restarts

    for restart in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET - polish_time - 3.0:
            break

        restart_deadline = time.time() + time_per_restart
        init_rng = random.Random(12345 + n + restart * 7919)

        # Greedy growth with varied temperature
        temperature = 0.3 + rng.random() * 1.2
        initial = _greedy_grow(sequence, init_rng, temperature)

        if initial is None:
            initial = _greedy_grow(sequence, init_rng, 0.3)
        if initial is None:
            initial = [(i, 0) for i in range(n)]

        # SA refinement for this restart
        sa_time_left = min(time_per_restart, max(0.5, restart_deadline - time.time()))
        if sa_time_left > 0.5:
            refined_chain, refined_energy = _pivot_sa(
                sequence, initial, time_limit=sa_time_left, rng=rng
            )
        else:
            refined_chain, refined_energy = initial, _energy(sequence, initial)

        if best_chain is None or refined_energy < best_energy:
            best_chain = list(refined_chain)
            best_energy = refined_energy
=======
    # Multi-restart: each restart gets greedy init + SA
    time_per_restart = sa_total_time / num_restarts

    for restart in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET - polish_time - 3.0:
            break

        restart_deadline = time.time() + time_per_restart
        init_rng = random.Random(12345 + n + restart * 7919)

        # Greedy growth with varied temperature
        temperature = 0.2 + rng.random() * 1.0
        initial = _greedy_grow(sequence, init_rng, temperature)

        if initial is None:
            initial = _greedy_grow(sequence, init_rng, 0.3)
        if initial is None:
            initial = [(i, 0) for i in range(n)]

        # SA refinement for this restart
        sa_time_left = min(time_per_restart, max(0.5, restart_deadline - time.time()))
        if sa_time_left > 0.5:
            refined_chain, refined_energy = _pivot_sa(
                sequence, initial, time_limit=sa_time_left, rng=rng
            )
        else:
            refined_chain, refined_energy = initial, _energy(sequence, initial)

        if best_chain is None or refined_energy < best_energy:
            best_chain = list(refined_chain)
            best_energy = refined_energy
>>>>>>> REPLACE

