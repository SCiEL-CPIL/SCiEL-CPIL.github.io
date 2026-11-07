    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 20s margin

        def safe_call(x):
            try:
                x = np.asarray(x, dtype=float)
                if not np.all(np.isfinite(x)):
                    return 1e10
                x = np.clip(x, -5.0, 5.0)
                val = float(func(x))
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        best_x = None
        best_f = np.inf
        calls_made = 0

        def track_call(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        try:
            # ... existing logic ...
        except Exception:
            pass

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            track_call(best_x, best_f)

        return best_x, best_f
    