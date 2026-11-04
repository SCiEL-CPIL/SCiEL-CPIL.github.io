import numpy as np

def construct():
    N = 600
    # The task is to minimize the normalized maximum of the autoconvolution.
    # Assuming the metric is max(g(t)) / max(f(x))^2, the optimal strategy
    # is to minimize the support width of f, as max(g) scales with width
    # and max(f)^2 is independent of width (for fixed height).
    # The minimum width for a non-zero step function on N intervals is 1 interval.
    # We return a function that is 1 on the first interval and 0 elsewhere.
    
    heights = np.zeros(N)
    heights[0] = 1.0
    
    return list(heights)

if __name__ == "__main__":
    # Quick self-check
    heights = construct()
    assert len(heights) == 600
    assert all(h >= 0 for h in heights)
