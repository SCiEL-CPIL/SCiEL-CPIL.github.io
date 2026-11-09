import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    def get_base_mask(u, v):
        # Base corridor:
        # Arm 1: 0 <= u <= 1, -3 <= v <= 1
        # Arm 2: -3 <= u <= 1, 0 <= v <= 1
        return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
               ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))

    def evaluate_path_dense(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        """
        # Prepend (0,0,0) and append (-2,0,90)
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path exactly as evaluator does
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
        
        valid = np.ones(n_points, dtype=bool)
        
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = get_base_mask(u, v)
            
            valid = valid & in_base
            
            if not np.any(valid):
                return 0.0
                
        return np.sum(valid) * 0.0001

    def evaluate_arc(radius, offset_y):
        """
        Evaluate a circular arc path.
        The arc starts at (0,0) with angle 0 and ends at (-2,0) with angle 90.
        The center of the circle is at (-radius, offset_y).
        The radius must be such that the distance from center to (0,0) is radius,
        and from center to (-2,0) is radius.
        Distance to (0,0): sqrt((-radius)^2 + offset_y^2) = radius
        => radius^2 + offset_y^2 = radius^2 => offset_y = 0? 
        Wait, if center is (cx, cy), dist to (0,0) is R, dist to (-2,0) is R.
        cx^2 + cy^2 = R^2
        (cx+2)^2 + cy^2 = R^2
        cx^2 + 4cx + 4 + cy^2 = cx^2 + cy^2
        4cx + 4 = 0 => cx = -1.
        So the center x-coordinate is fixed at -1.
        Then R^2 = 1 + cy^2.
        So the only free parameter is cy (offset_y).
        Let's verify the angle change.
        Vector from center to start (0,0): (1, -cy). Angle = atan2(-cy, 1).
        Vector from center to end (-2,0): (-1, -cy). Angle = atan2(-cy, -1).
        We need the rotation to be 90 degrees (pi/2).
        If cy > 0, start angle is negative (4th quad), end angle is 3rd quad.
        Let's check the sweep.
        Start vector (1, -cy). End vector (-1, -cy).
        Dot product: -1 + cy^2.
        Cross product (2D): 1*(-cy) - (-cy)*(-1) = -cy - cy = -2cy.
        If cy > 0, cross product is negative, so clockwise rotation?
        The problem defines rotation a.
        At start, a=0. The corridor is aligned with x-axis.
        The vector from center to the "inner corner" (which is at the placement origin) is what rotates.
        Wait, the placement is (x, y, a). The inner corner is at (x,y).
        So the vector from center to (x,y) rotates by a.
        If we move along a circle centered at (-1, cy) with radius R,
        the vector from center to point (x,y) rotates.
        Does it rotate by 90 degrees?
        Start: (0,0). Vector from (-1, cy) is (1, -cy).
        End: (-2,0). Vector from (-1, cy) is (-1, -cy).
        Angle of start vector: theta1 = atan2(-cy, 1).
        Angle of end vector: theta2 = atan2(-cy, -1).
        We want theta2 - theta1 = 90 degrees (or -90).
        Let's test cy = 1.
        theta1 = atan2(-1, 1) = -45 deg.
        theta2 = atan2(-1, -1) = -135 deg.
        Difference: -90 deg.
        This corresponds to a rotation of -90 degrees.
        The evaluator expects a rotation from 0 to 90 degrees.
        So we need the angle to increase by 90.
        This implies we need to traverse the circle in the other direction or choose cy such that angles increase.
        If cy < 0, say cy = -1.
        theta1 = atan2(1, 1) = 45 deg.
        theta2 = atan2(1, -1) = 135 deg.
        Difference: 90 deg.
        So we need cy < 0.
        Let cy = -h where h > 0.
        Center is (-1, -h).
        R = sqrt(1 + h^2).
        The path is the arc from (0,0) to (-2,0) passing "above" the center?
        No, if center is (-1, -h), and h>0, center is below x-axis.
        Points (0,0) and (-2,0) are above the center.
        The arc connecting them that subtends 90 degrees at the center...
        Wait, the angle subtended at the center is the change in the vector from center to point.
        Yes, that is the rotation a.
        So we parameterize by h > 0.
        Center: (-1, -h).
        Radius: R = sqrt(1 + h^2).
        Start angle (vector from center to (0,0)): phi_start = atan2(0 - (-h), 0 - (-1)) = atan2(h, 1).
        End angle (vector from center to (-2,0)): phi_end = atan2(0 - (-h), -2 - (-1)) = atan2(h, -1).
        We need phi_end - phi_start = pi/2 (90 deg).
        Let's check.
        phi_start = arctan(h).
        phi_end = pi - arctan(h).
        Difference = pi - 2*arctan(h).
        We want this to be pi/2.
        pi - 2*arctan(h) = pi/2 => 2*arctan(h) = pi/2 => arctan(h) = pi/4 => h = 1.
        So ONLY h=1 works for a pure circular arc where the center is fixed at x=-1?
        Let's re-read the geometry.
        The path is a curve. The rotation a(t) is determined by the tangent? No, "rotated by a degrees".
        The task says: "Row (x, y, a) places... rotated by a degrees."
        The evaluator interpolates (x, y, a).
        It does NOT derive a from the path geometry. We supply a.
        However, for a "natural" motion, a should correspond to the orientation of the corridor.
        If we use a circular arc for the position (x(t), y(t)), we can choose a(t) independently?
        No, usually "moving sofa" implies the sofa rotates as it moves.
        The inspiration says: "Model the path as a continuous curve where the corridor centerline follows a circular arc... parameterized by radius and center offset... ensures smooth rotation and translation simultaneously".
        This implies that the rotation a is linked to the position on the arc.
        If we constrain a to be the angle of the vector from the center to the current position, then a is determined by the geometry.
        But we saw that for the endpoints (0,0) and (-2,0) to have a difference of 90 degrees with a circle centered on x=-1, we MUST have h=1 (R=sqrt(2)).
        This leaves NO degrees of freedom! The path is fixed.
        
        Is there another circular arc?
        What if the center is not on x=-1?
        We established cx = -1 from the condition that dist(center, start) = dist(center, end) = R.
        Is it required that the start and end are on the circle?
        Yes, the path starts at (0,0) and ends at (-2,0).
        If the path is a circular arc, the start and end points must lie on the circle.
        And the rotation must change by 90 degrees.
        If the rotation a is defined as the angle of the vector from center to point, then the change in a is the angle subtended by the arc at the center.
        So we need an arc from (0,0) to (-2,0) that subtends 90 degrees at the center.
        Locus of points from which the segment (0,0)-(-2,0) subtends 90 degrees is the circle with diameter (0,0)-(-2,0)?
        No, Thales theorem: angle in a semicircle is 90.
        So if the center is the midpoint (-1, 0), the angle is 180? No.
        If the center is C, and vectors CS and CE are perpendicular.
        This means C lies on the circle with diameter SE? No.
        If angle SCE = 90, then C lies on the circle with diameter... no.
        The set of points C such that angle SCE = 90 is the circle with diameter SE?
        Let's check. If C is on circle with diameter SE, then angle SCE is 90?
        No, angle in semicircle is 90 means if S, C, E are on circle and SE is diameter, then angle SCE = 90.
        Here S=(0,0), E=(-2,0). Midpoint (-1,0). Radius 1.
        Circle: (x+1)^2 + y^2 = 1.
        If C is on this circle, then triangle SCE is right-angled at C.
        So vectors CS and CE are perpendicular.
        This means the angle subtended is 90 degrees.
        So the center C can be ANY point on the circle (x+1)^2 + y^2 = 1.
        Let C = (-1 + cos(theta), sin(theta)).
        Then R = dist(C, S) = sqrt( (cos(theta))^2 + (sin(theta))^2 ) = 1.
        So the radius is always 1!
        The path is a quarter circle of radius 1.
        The center moves around the unit circle centered at (-1,0).
        Let's parameterize by theta (angle of center around (-1,0)).
        Center C = (-1 + cos(theta), sin(theta)).
        Radius R = 1.
        Start point S=(0,0). Vector CS = (1-cos(theta), -sin(theta)).
        End point E=(-2,0). Vector CE = (-1-cos(theta), -sin(theta)).
        We need to traverse the arc from S to E.
        Which way? We need the rotation a to go from 0 to 90.
        Let's check the angle of CS.
        a_start = atan2(-sin(theta), 1-cos(theta)).
        a_end = atan2(-sin(theta), -1-cos(theta)).
        We want a_end - a_start = 90 deg (pi/2).
        Let's test theta = pi/2 (Center at (-1, 1)).
        CS = (1, -1). a_start = atan2(-1, 1) = -45 deg.
        CE = (-1, -1). a_end = atan2(-1, -1) = -135 deg.
        Diff = -90 deg. (Wrong direction).
        Let's test theta = -pi/2 (Center at (-1, -1)).
        CS = (1, 1). a_start = atan2(1, 1) = 45 deg.
        CE = (-1, 1). a_end = atan2(1, -1) = 135 deg.
        Diff = 90 deg. (Correct direction).
        So we need the center to be in the lower half plane (y < 0) relative to the chord?
        Actually, any theta in (pi, 2pi) or (-pi, 0) might work depending on the arc traversal.
        Let's parameterize the path.
        Angle of point P on circle relative to center C.
        Let phi be the angle of the vector from C to P.
        At S, phi_start = atan2(0-cy, 0-cx) = atan2(-sin(theta), 1-cos(theta)).
        At E, phi_end = atan2(0-cy, -2-cx) = atan2(-sin(theta), -1-cos(theta)).
        We established that for theta = -pi/2, phi_start = 45, phi_end = 135.
        So we traverse from 45 to 135 degrees.
        This is a 90 degree sweep.
        So the path is defined by the choice of theta.
        But wait, if R=1 is fixed, is there any other parameter?
        The shape of the arc is fixed (quarter circle of radius 1).
        Only the position of the arc (determined by theta) changes.
        So we have 1 parameter: theta.
        Let's verify if this is sufficient.
        The "inspiration" mentioned "radius and center offset".
        Maybe the radius is not fixed?
        If the angle subtended is NOT 90 degrees, then a(t) cannot be simply the geometric angle.
        But the task says "rotated by a degrees".
        If we use a circular arc with radius R != 1, the angle subtended is not 90.
        Then we would have to "stretch" the rotation a(t) to go from 0 to 90 while the position goes along the arc.
        This decouples a from the geometric tangent.
        Is that allowed?
        "Row (x, y, a) places... rotated by a degrees."
        The evaluator interpolates a.
        It does not check if a matches the tangent of the path.
        So we CAN use a circular arc of any radius R, and just linearly interpolate a from 0 to 90 over the path.
        This gives us 2 parameters: R and theta (or center position).
        Actually, if we fix the chord length (2) and the sweep of a (90), and we want the path to be a circular arc...
        The geometric sweep of the arc (angle at center) is 2*arcsin(chord/(2R)) = 2*arcsin(1/R).
        If we linearly map a (0 to 90) to the arc parameter t (0 to 1), the "rotation speed" vs "translation speed" will vary depending on R.
        
        So the strategy:
        1. Define a circular arc from (0,0) to (-2,0) with radius R.
        2. The center C is determined by R and a sign (upper/lower arc).
           C_x = -1.
           C_y = ±sqrt(R^2 - 1).
        3. We have 2 parameters: R and sign (or just C_y).
        4. We sample the path at integer degrees 1..89.
        5. For each integer degree a_k, we need to find the position (x_k, y_k) on the arc.
           How to map a_k to position?
           Option A: Linear in arc length.
           Option B: Linear in angle at center.
           Option C: Linear in x.
           The evaluator interpolates (x,y,a) linearly between our output points.
           If we output points at a=1, 2, ..., 89, the evaluator will just use them directly (m=1).
           So we just need to choose (x_k, y_k) for each a_k.
           A natural choice is to map a_k linearly to the geometric angle phi on the circle.
           phi_start = angle of S relative to C.
           phi_end = angle of E relative to C.
           phi_k = phi_start + (phi_end - phi_start) * (k/90).
           Then x_k = C_x + R*cos(phi_k), y_k = C_y + R*sin(phi_k).
           
        Let's implement this.
        Parameters: R (radius), s (sign of C_y, +1 or -1).
        Actually, C_y = s * sqrt(R^2 - 1).
        We need R >= 1.
        
        Let's search over R and s.
        
    """
    def evaluate_arc_params(R, s):
        if time.time() > deadline - 5:
            return 0.0
            
        if R < 1.0:
            return 0.0
            
        try:
            cy = s * math.sqrt(R*R - 1.0)
            cx = -1.0
        except ValueError:
            return 0.0
            
        # Calculate start and end angles relative to center
        # S = (0,0), E = (-2,0)
        # Vector CS = (1, -cy)
        # Vector CE = (-1, -cy)
        phi_start = math.atan2(-cy, 1.0)
        phi_end = math.atan2(-cy, -1.0)
        
        # Ensure we traverse the short arc (90 deg of 'a' should map to the arc)
        # We want the difference phi_end - phi_start to be consistent with the direction of 'a' (0 to 90).
        # If s=1 (center above), cy>0.
        # phi_start = atan2(-cy, 1) (4th quad, negative)
        # phi_end = atan2(-cy, -1) (3rd quad, negative, more negative)
        # Diff is negative.
        # If s=-1 (center below), cy<0.
        # phi_start = atan2(-cy, 1) (1st quad, positive)
        # phi_end = atan2(-cy, -1) (2nd quad, positive, larger)
        # Diff is positive.
        # Since a goes 0->90 (positive), we prefer the mapping where phi increases.
        # So s=-1 is the natural one. But let's allow both and see.
        
        # Map a in [0, 90] to phi in [phi_start, phi_end]
        # We output points for a = 1..89.
        angles = np.arange(1, 90)
        t = angles / 90.0
        
        phis = phi_start + t * (phi_end - phi_start)
        
        xs_arc = cx + R * np.cos(phis)
        ys_arc = cy + R * np.sin(phis)
        
        path = np.column_stack((xs_arc, ys_arc, angles))
        
        return evaluate_path_dense(path)

    # Search over R and s
    # R range: 1.0 to 5.0
    # s range: -1, 1
    
    best_score = -np.inf
    best_params = None
    
    # Coarse grid
    R_vals = np.arange(1.0, 5.0, 0.1)
    s_vals = [-1.0, 1.0]
    
    for s in s_vals:
        for R in R_vals:
            score = evaluate_arc_params(R, s)
            if score > best_score:
                best_score = score
                best_params = (R, s)
                
    # Fine grid around best
    if best_params is not None:
        R_best, s_best = best_params
        R_fine = np.arange(max(1.0, R_best - 0.5), R_best + 0.5, 0.01)
        for R in R_fine:
            score = evaluate_arc_params(R, s_best)
            if score > best_score:
                best_score = score
                best_params = (R, s_best)
                
    # Construct final path
    R, s = best_params
    cy = s * math.sqrt(R*R - 1.0)
    cx = -1.0
    
    phi_start = math.atan2(-cy, 1.0)
    phi_end = math.atan2(-cy, -1.0)
    
    angles = np.arange(1, 90)
    t = angles / 90.0
    phis = phi_start + t * (phi_end - phi_start)
    
    xs_arc = cx + R * np.cos(phis)
    ys_arc = cy + R * np.sin(phis)
    
    final_path = np.column_stack((xs_arc, ys_arc, angles))
    
    return final_path
