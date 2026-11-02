<<<<<<< SEARCH
    def overlap_penalty(points):
        # Sum of squared penetration depths
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                diff = points[i] - points[j]
                projs = np.abs(diff @ normals.T)
                max_proj = np.max(projs)
                if max_proj < MINKOWSKI_APOTHEM:
                    pen = MINKOWSKI_APOTHEM - max_proj
                    penalty += pen * pen
        return penalty
=======
    def overlap_penalty(points):
        # Sum of squared violations of the exact distance constraint
        penalty = 0.0
        h = APOTHEM
        for i in range(N):
            for j in range(i+1, N):
                v = points[i] - points[j]
                D = np.linalg.norm(v)
                if D < 1e-9:
                    # Centers coincide, maximum overlap
                    penalty += (2 * h) ** 2
                    continue
                theta = np.arctan2(v[1], v[0])
                # Map theta to the range [0, pi/3) to find the sector
                t_mod = theta % (math.pi / 3.0)
                # The normal to the side in this sector is at pi/6
                # The angle difference from the normal
                delta = t_mod - math.pi / 6.0
                cos_delta = math.cos(delta)
                # The distance from center to boundary in direction theta is h / cos_delta
                # For two hexagons to be disjoint, D >= 2 * (h / cos_delta)
                # => D * cos_delta >= 2 * h
                val = D * cos_delta
                if val < 2 * h:
                    pen = 2 * h - val
                    penalty += pen * pen
        return penalty
>>>>>>> REPLACE

Also, to be safe against the optimizer stopping with a small but non-zero penalty, we can increase the penalty weight in the objective function or add a final check.
The current objective is `S + 1000.0 * pen`.
If `pen` is small, `S` dominates. But if `pen` is not zero, the configuration is invalid.
We should probably ensure that if `pen > 1e-6`, we treat it as a large error.
The current code does `if pen > 1e-6: return S + 1000.0 * pen`.
This is okay, but maybe the weight 1000 is not enough if `S` is large? No, `S` is around 3-4. `pen` is squared distance. If overlap is 0.1, `pen` is 0.01, `1000*0.01 = 10`. `S+10` is much larger than valid `S`. So it should push away.
However, if the optimizer gets stuck in a local minimum with `pen > 0`, it returns an invalid solution.
To mitigate this, we can try multiple restarts or ensure the initial points are non-overlapping.
The "Flower" configuration has hexagons touching, so `pen` should be 0 (or very close).
The "Grid" configuration has hexagons touching or overlapping depending on the offset.
In the grid:
Row 0: (0,0), (sqrt(3),0), ...
Row 1: (sqrt(3)/2, 1.5), ...
Distance between (0,0) and (sqrt(3)/2, 1.5):
dx = sqrt(3)/2, dy = 1.5.
D^2 = 3/4 + 9/4 = 12/4 = 3. D = sqrt(3) ≈ 1.732.
Required D for theta:
v = (-sqrt(3)/2, -1.5) (from row 1 to row 0).
theta = atan2(-1.5, -sqrt(3)/2).
This is in the 3rd quadrant.
Angle is approx -120 deg or 240 deg.
t_mod = 240 % 60 = 0 deg.
delta = 0 - 30 = -30 deg.
cos(-30) = sqrt(3)/2.
Req D = 2 * h / cos(30) = 2 * (sqrt(3)/2) / (sqrt(3)/2) = 2.
Actual D = sqrt(3) ≈ 1.732.
So they overlap!
The grid initialization is invalid.
The random initialization is likely invalid.
Only the Flower configuration is valid (touching).
Starting from invalid points with a penalty method can be tricky if the penalty landscape is not convex or has local minima.
However, SLSQP with a quadratic penalty usually handles this by moving points apart.

To improve robustness, we can add a "separation" step before optimization or use a higher penalty weight.
Let's increase the penalty weight to `10000.0` to be safer.

And, we should ensure that if the best found solution still has `pen > 1e-5`, we might want to try to repair it or just return the best valid one found. Since we don't track "validity" separately from the objective, and the objective includes the penalty, the minimum of the objective should ideally be at `pen=0`.

Let's update the penalty weight.

