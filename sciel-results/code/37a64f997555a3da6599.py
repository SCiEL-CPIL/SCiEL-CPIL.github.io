    def _line_search(pts, grad, lr_init=0.1, max_iter=20):
        # Simple backtracking line search
        # We don't have a scalar energy function easily available (hull changes),
        # so we use the max circumradius as a proxy, but it's non-smooth.
        # Instead, we can just use a fixed schedule or monitor the change in points.
        # For simplicity and robustness, we'll stick to adaptive lr based on improvement.
        pass
    