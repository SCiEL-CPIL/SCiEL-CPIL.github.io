import numpy as np
import time

N = 11
GAP = 1.001  # Slight expansion to ensure disjointness is recognized

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    sqrt3 = np.sqrt(3.0)

    def hex_vertices(cx, cy, rot, scale=1.0):
        base = np.array([[np.cos(np.pi/3*k), np.sin(np.pi/3*k)] for k in range(6)])
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        verts = (base * scale) @ R.T + np.array([cx, cy])
        return verts

    def min_side_for_vertices(all_verts, C, theta):
        """Minimum outer hex side length to contain all vertices."""
        normals = np.array([[np.cos(theta + k*np.pi/3), np.sin(theta + k*np.pi/3)] for k in range(6)])
        diffs = all_verts - C
        dots = diffs @ normals.T
        max_dot = np.max(dots)
        return max_dot / (sqrt3 / 2.0)

    def check_disjoint(centers, rotations, scales):
        """Check if hexagons are pairwise disjoint (with gap)."""
        for i in range(len(centers)):
            vi = hex_vertices(centers[i][0], centers[i][1], rotations[i], scales[i])
            for j in range(i+1, len(centers)):
                vj = hex_vertices(centers[j][0], centers[j][1], rotations[j], scales[j])
                # Check if any edge of i intersects any edge of j
                # Simplified: check if any vertex of i is inside j or vice versa
                # For regular hexagons, if they don't share edges, checking vertex containment suffices
                for v in vi:
                    if _point_in_hex(v, vj):
                        return False
                for v in vj:
                    if _point_in_hex(v, vi):
                        return False
        return True

    def _point_in_hex(p, hex_verts):
        """Check if point p is inside hexagon with vertices hex_verts."""
        # Use cross product method for convex polygon
        for k in range(6):
            a = hex_verts[k]
            b = hex_verts[(k+1) % 6]
            cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
            if cross < -1e-10:
                return False
        return True

    # Build compact cluster: rows 2-3-3-2-1
    v_spacing = sqrt3
    h_spacing = 2.0

    def make_cluster_config():
        centers = []
        # Row 0: 2
        for i in range(2):
            x = (i - 0.5) * h_spacing
            centers.append((x, 2.0 * v_spacing))
        # Row 1: 3
        for i in range(3):
            x = (i - 1.0) * h_spacing
            centers.append((x, 1.0 * v_spacing))
        # Row 2: 3
        for i in range(3):
            x = (i - 1.0) * h_spacing
            centers.append((x, 0.0))
        # Row 3: 2
        for i in range(2):
            x = (i - 0.5) * h_spacing
            centers.append((x, -1.0 * v_spacing))
        # Row 4: 1
        centers.append((0.0, -2.0 * v_spacing))
        return centers

    def make_cluster_config_B():
        centers = []
        # Row 0: 3
        for i in range(3):
            x = (i - 1.0) * h_spacing
            centers.append((x, 2.0 * v_spacing))
        # Row 1: 2
        for i in range(2):
            x = (i - 0.5) * h_spacing
            centers.append((x, 1.0 * v_spacing))
        # Row 2: 3
        for i in range(3):
            x = (i - 1.0) * h_spacing
            centers.append((x, 0.0))
        # Row 3: 2
        for i in range(2):
            x = (i - 0.5) * h_spacing
            centers.append((x, -1.0 * v_spacing))
        # Row 4: 1
        centers.append((0.0, -2.0 * v_spacing))
        return centers

    def make_cluster_config_C():
        centers = []
        # Row 0: 1
        centers.append((0.0, 2.0 * v_spacing))
        # Row 1: 3
        for i in range(3):
            x = (i - 1.0) * h_spacing
            centers.append((x, 1.0 * v_spacing))
        # Row 2: 3
        for i in range(3):
            x = (i - 1.0) * h_spacing
            centers.append((x, 0.0))
        # Row 3: 3
        for i in range(3):
            x = (i - 1.0) * h_spacing
            centers.append((x, -1.0 * v_spacing))
        # Row 4: 1
        centers.append((0.0, -2.0 * v_spacing))
        return centers

    configs = [make_cluster_config(), make_cluster_config_B(), make_cluster_config_C()]
    rotations = [0.0] * N
    scales = [1.0] * N

    best_result = None
    best_s = np.inf

    for config_idx, centers in enumerate(configs):
        if time.time() - t0 > DEADLINE - 15:
            break

        # Collect all vertices (with GAP for disjointness safety)
        all_verts = []
        for idx, (cx, cy) in enumerate(centers):
            verts = hex_vertices(cx, cy, rotations[idx], scales[idx] * GAP)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)

        # Optimize outer hexagon
        cluster_center = np.mean(centers, axis=0)

        # Coarse theta search
        best_theta = 0.0
        best_s_local = np.inf
        for th in np.linspace(0, np.pi/3, 25):
            s = min_side_for_vertices(all_verts, cluster_center, th)
            if s < best_s_local:
                best_s_local = s
                best_theta = th

        # Coarse C search
        search_range = 5.0
        n_grid = 21
        cx_vals = np.linspace(cluster_center[0] - search_range, cluster_center[0] + search_range, n_grid)
        cy_vals = np.linspace(cluster_center[1] - search_range, cluster_center[1] + search_range, n_grid)

        best_C = cluster_center.copy()
        best_s_c = best_s_local
        for cx in cx_vals:
            for cy in cy_vals:
                C = np.array([cx, cy])
                s = min_side_for_vertices(all_verts, C, best_theta)
                if s < best_s_c:
                    best_s_c = s
                    best_C = C.copy()

        # Refine
        for refinement in range(4):
            step = search_range / (n_grid * (refinement + 1) * 2)
            cx_fine = np.linspace(best_C[0] - 3*step, best_C[0] + 3*step, 31)
            cy_fine = np.linspace(best_C[1] - 3*step, best_C[1] + 3*step, 31)
            for cx in cx_fine:
                for cy in cy_fine:
                    C = np.array([cx, cy])
                    s = min_side_for_vertices(all_verts, C, best_theta)
                    if s < best_s_c:
                        best_s_c = s
                        best_C = C.copy()

            # Refine theta
            th_step = (np.pi/3) / (25 * (refinement + 1) * 2)
            th_lo = max(0, best_theta - 3*th_step)
            th_hi = min(np.pi/3, best_theta + 3*th_step)
            th_fine = np.linspace(th_lo, th_hi, 31)
            for th in th_fine:
                s = min_side_for_vertices(all_verts, best_C, th)
                if s < best_s_c:
                    best_s_c = s
                    best_theta = th

            if time.time() - t0 > DEADLINE - 10:
                break

        if best_s_c < best_s:
            best_s = best_s_c
            best_result = (centers, best_C, best_theta)

    if best_result is None:
        # Fallback
        centers = make_cluster_config()
        all_verts = []
        for cx, cy in centers:
            verts = hex_vertices(cx, cy, 0.0, GAP)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        C = np.array([0.0, 0.0])
        s = min_side_for_vertices(all_verts, C, 0.0)
        return [[float(cx), float(cy), 0.0] for cx, cy in centers], (0.0, 0.0), float(s), 0.0

    centers, C, theta = best_result
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
    return inner, (float(C[0]), float(C[1])), float(best_s), float(np.degrees(theta))
