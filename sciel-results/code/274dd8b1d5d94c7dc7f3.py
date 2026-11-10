        if best_paths is not None:
            # Validate
            valid = all(is_valid_path(p, adj, n) for p in best_paths)
            if valid:
                return best_paths
        return [[u, v] for u, v in edges]
