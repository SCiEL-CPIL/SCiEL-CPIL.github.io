import numpy as np

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    points = np.zeros((n, 2))
    
    # 16 points on a "cap" (concave down) near the top
    # y = 0.9 - 0.05 * x^2
    # x in [0.05, 0.95]
    x_top = np.linspace(0.05, 0.95, 16)
    y_top = 0.9 - 0.05 * x_top**2
    points[:16] = np.column_stack((x_top, y_top))
    
    # 16 points on a "cup" (concave up) near the bottom
    # y = 0.1 + 0.05 * x^2
    x_bot = np.linspace(0.05, 0.95, 16)
    y_bot = 0.1 + 0.05 * x_bot**2
    points[16:32] = np.column_stack((x_bot, y_bot))
    
    # 1 point in the center
    points[32] = [0.5, 0.5]
    
    return points
