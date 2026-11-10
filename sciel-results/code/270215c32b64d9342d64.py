<<<<<<< SEARCH
        best_pieces = None
        best_count = len(edge_set) + 1

        # Try multiple strategies
        strategies = [
            ("greedy_longest", _greedy_longest_cycles),
            ("matching_paths", _matching_based_cycles),
        ]

        for strategy_name, strategy_func in strategies:
            if time.time() > deadline - 20:
                break
            try:
                pieces = strategy_func(edge_set, adj, deadline)
                if pieces is not None:
                    count = len(pieces)
                    if count < best_count:
                        best_count = count
                        best_pieces = [p[:] for p in pieces]
            except Exception:
                pass

        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]

        return best_pieces
=======
        best_pieces = None
        best_count = len(edge_set) + 1

        def is_valid_decomposition(pieces, edge_set):
            covered = set()
            for piece in pieces:
                if len(piece) == 2:
                    e = tuple(sorted(piece))
                    if e in covered:
                        return False
                    if e not in edge_set:
                        return False
                    covered.add(e)
                else:
                    # Cycle
                    if len(piece) < 3:
                        return False
                    # Check distinct vertices
                    if len(set(piece)) != len(piece):
                        return False
                    for i in range(len(piece)):
                        u, v = piece[i], piece[(i + 1) % len(piece)]
                        e = tuple(sorted((u, v)))
                        if e in covered:
                            return False
                        if e not in edge_set:
                            return False
                        covered.add(e)
            return covered == edge_set

        # Try multiple strategies
        strategies = [
            ("greedy_longest", _greedy_longest_cycles),
        ]

        for strategy_name, strategy_func in strategies:
            if time.time() > deadline - 20:
                break
            try:
                pieces = strategy_func(edge_set, adj, deadline)
                if pieces is not None and is_valid_decomposition(pieces, edge_set):
                    count = len(pieces)
                    if count < best_count:
                        best_count = count
                        best_pieces = [p[:] for p in pieces]
            except Exception:
                pass

        if best_pieces is None:
            best_pieces = [list(e) for e in edge_set]

        return best_pieces
>>>>>>> REPLACE
