<<<<<<< SEARCH
def _check_valid(pts):
    """Check all constraints for a set of integer points."""
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    # Check all pairwise distances are integers
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts[i], pts[j], pts[k], pts[l]
                    a11 = x1 - x2
                    a12 = y1 - y2
                    b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                    a21 = x1 - x3
                    a22 = y1 - y3
                    b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                    det = a11 * a22 - a21 * a12
                    if det == 0:
                        continue
                    A_num = b1 * a22 - b2 * a12
                    B_num = a11 * b2 - a21 * b1
                    C_num = -(x1*x1 + y1*y1) * det - x1 * A_num - y1 * B_num
                    lhs = x4 * A_num + y4 * B_num + C_num
                    rhs = -(x4*x4 + y4*y4) * det
                    if lhs == rhs:
                        return False
    return True
=======
def _check_valid(pts):
    """Check all constraints for a set of integer points."""
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    # Check all pairwise distances are integers
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts[i], pts[j], pts[k], pts[l]
                    a11 = x1 - x2
                    a12 = y1 - y2
                    b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                    a21 = x1 - x3
                    a22 = y1 - y3
                    b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                    det = a11 * a22 - a21 * a12
                    if det == 0:
                        continue
                    A_num = b1 * a22 - b2 * a12
                    B_num = a11 * b2 - a21 * b1
                    C_num = -(x1*x1 + y1*y1) * det - x1 * A_num - y1 * B_num
                    lhs = x4 * A_num + y4 * B_num + C_num
                    rhs = -(x4*x4 + y4*y4) * det
                    if lhs == rhs:
                        return False
    return True

def _check_new_point(pts, new_pt):
    """Check if adding new_pt to pts maintains all constraints (incremental)."""
    x_new, y_new = new_pt
    n = len(pts)
    # Check distance to all existing points
    for i in range(n):
        dx = pts[i][0] - x_new
        dy = pts[i][1] - y_new
        d2 = dx * dx + dy * dy
        if not _is_square(d2):
            return False
    # Check no three collinear (new point with any pair)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            if (x2 - x1) * (y_new - y1) - (x_new - x1) * (y2 - y1) == 0:
                return False
    # Check no four concyclic (new point with any triple)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                (x1, y1), (x2, y2), (x3, y3) = pts[i], pts[j], pts[k]
                a11 = x1 - x2
                a12 = y1 - y2
                b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                a21 = x1 - x3
                a22 = y1 - y3
                b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                det = a11 * a22 - a21 * a12
                if det == 0:
                    continue
                A_num = b1 * a22 - b2 * a12
                B_num = a11 * b2 - a21 * b1
                C_num = -(x1*x1 + y1*y1) * det - x1 * A_num - y1 * B_num
                lhs = x_new * A_num + y_new * B_num + C_num
                rhs = -(x_new*x_new + y_new*y_new) * det
                if lhs == rhs:
                    return False
    return True
>>>>>>> REPLACE

