<<<<<<< SEARCH
def _local_search(sequence, chain, max_iterations=2000, time_limit=5.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.997

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Try a pivot or crankshaft move
        if rng.random() < 0.7:
            new_chain = _pivot_move(current_chain, rng)
        else:
            new_chain = _crankshaft_move(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain
=======
def _local_search(sequence, chain, max_iterations=2000, time_limit=5.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.997

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Try a pivot or crankshaft move
        if rng.random() < 0.7:
            new_chain = _pivot_move(current_chain, rng)
        else:
            new_chain = _crankshaft_move(current_chain, rng)

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature *= cooling

    return best_chain
>>>>>>> REPLACE

The local search looks fine. Now let me also update `_pivot_move` and `_crankshaft_move` to use the list-returning `_neighbours` if needed. Looking at the code, they don't use `_neighbours` directly, so they should be fine.

Let me also make sure the `_greedy_fallback` function is properly defined. It should be:

