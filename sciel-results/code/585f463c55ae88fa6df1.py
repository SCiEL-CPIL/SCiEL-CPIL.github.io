<<<<<<< SEARCH
def construct(instance):
    """Best of hybrid search on Dzugutov clusters."""
    size = instance["n"]
    rng = np.random.default_rng(size * 1000 + 42)

    # Time budget per instance
    remaining = _DEADLINE - time.time()
    time_budget = max(5.0, min(45.0, remaining / 1.5))

    result = _hybrid_search(size, rng, time_budget=time_budget)
    return result
=======
def construct(instance):
    """Best of hybrid search on Dzugutov clusters."""
    size = instance["n"]
    rng = np.random.default_rng(size * 1000 + 42)

    # Time budget per instance
    remaining = _DEADLINE - time.time()
    time_budget = max(5.0, min(45.0, remaining / 1.5))

    result = _hybrid_search(size, rng, time_budget=time_budget)
    result = np.array(result, dtype=float)
    if result.shape != (size, 3):
        # Fallback: random compact
        result = _random_compact(size, rng, spacing=1.05)
    return result
>>>>>>> REPLACE
