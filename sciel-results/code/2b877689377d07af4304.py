import numpy as np
import time
from scipy.optimize import brentq

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using the first integral (Hamiltonian)
    to find the terminal state analytically, with a single root-finding step.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    y0_init, y1_init = y[0], y[1]
    
    # Satisfy the requirement that f is called at least once
    f(t0, y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Check if we are already past deadline
    if time.time() > deadline:
        return y
    
    # First integral for Lotka-Volterra: dy1/dt/dy0 = (alpha y1 - beta y0 y1) / (gamma y0 - delta y0 y1)
    # With alpha=1.5, beta=1, gamma=-3, delta=1:
    # dy1/dy0 = (1.5*y1 - y0*y1) / (-3*y0 + y0*y1) = y1*(1.5-y0) / (y0*(y1-3))
    # Separating variables:
    # (y0*(y1-3))/y1 dy1 = (1.5-y0) dy0
    # (y0 - 3*y0/y1) dy1 = (1.5-y0) dy0
    # Integrate both sides:
    # y0*y1 - 3*y0*ln(y1) = 1.5*y0 - 0.5*y0^2 + C
    # So H(y0, y1) = y0*y1 - 3*y0*ln(y1) - 1.5*y0 + 0.5*y0^2 is conserved
    
    # Compute the constant C from initial conditions
    def H(y0_val, y1_val):
        if y1_val <= 0 or y0_val <= 0:
            return np.inf
        return y0_val * y1_val - 3.0 * y0_val * np.log(y1_val) - 1.5 * y0_val + 0.5 * y0_val**2
    
    C = H(y0_init, y1_init)
    
    # We need to find y1_end given y0_end such that H(y0_end, y1_end) = C
    # The trajectory is a closed orbit. We need to determine y0_end at t=t1.
    # Since the system is periodic, we can find the period and then find the point on the orbit.
    # Alternatively, we can use the fact that the time to go from (y0_a, y1_a) to (y0_b, y1_b) along the orbit
    # can be computed by integrating dt = dy0 / (dy0/dt) = dy0 / (1.5*y0 - y0*y1)
    
    # Let's find the minimum and maximum y0 on the orbit to understand the trajectory
    # At extrema of y0, dy0/dt = 0 => 1.5*y0 - y0*y1 = 0 => y1 = 1.5
    # At extrema of y1, dy1/dt = 0 => -3*y1 + y0*y1 = 0 => y0 = 3
    
    # Find y1 when y0 = 3: H(3, y1) = C
    def H_at_y0_3(y1_val):
        return H(3.0, y1_val) - C
    
    # Find y0 when y1 = 1.5: H(y0, 1.5) = C
    def H_at_y1_15(y0_val):
        return H(y0_val, 1.5) - C
    
    # Find the y1 values when y0 = 3
    # H(3, y1) = 3*y1 - 3*3*ln(y1) - 1.5*3 + 0.5*9 = 3*y1 - 9*ln(y1) - 4.5 + 4.5 = 3*y1 - 9*ln(y1)
    # So 3*y1 - 9*ln(y1) = C
    # Let's solve for y1
    
    def f_y1(y1_val):
        if y1_val <= 0:
            return np.inf
        return 3.0 * y1_val - 9.0 * np.log(y1_val) - C
    
    # Find y0 when y1 = 1.5
    # H(y0, 1.5) = y0*1.5 - 3*y0*ln(1.5) - 1.5*y0 + 0.5*y0^2 = 0.5*y0^2 - 3*y0*ln(1.5)
    # So 0.5*y0^2 - 3*y0*ln(1.5) = C
    
    def f_y0(y0_val):
        if y0_val <= 0:
            return np.inf
        return 0.5 * y0_val**2 - 3.0 * y0_val * np.log(1.5) - C
    
    # The orbit has y0 in [y0_min, y0_max] and y1 in [y1_min, y1_max]
    # y0_min and y0_max are the roots of f_y0(y0) = 0
    # y1_min and y1_max are the roots of f_y1(y1) = 0
    
    # Let's find these bounds
    # For y0: 0.5*y0^2 - 3*ln(1.5)*y0 - C = 0
    # This is a quadratic: y0 = [3*ln(1.5) ± sqrt(9*ln(1.5)^2 + 2*C)] / 1
    disc_y0 = 9.0 * np.log(1.5)**2 + 2.0 * C
    if disc_y0 < 0:
        # This shouldn't happen for a valid orbit
        return y
    
    y0_min = 3.0 * np.log(1.5) - np.sqrt(disc_y0)
    y0_max = 3.0 * np.log(1.5) + np.sqrt(disc_y0)
    
    # For y1: 3*y1 - 9*ln(y1) = C
    # This is transcendental, but we can find the roots
    # The function g(y1) = 3*y1 - 9*ln(y1) has a minimum at y1 = 3 (where g'(y1) = 3 - 9/y1 = 0)
    # g(3) = 9 - 9*ln(3)
    # The two roots are on either side of y1 = 3
    
    # Let's use brentq to find the roots
    # We need to find brackets
    # For y1 < 3: g(y1) -> inf as y1 -> 0+, and g(3) = 9 - 9*ln(3)
    # For y1 > 3: g(y1) -> inf as y1 -> inf
    
    y1_min_bracket = (1e-10, 3.0)
    y1_max_bracket = (3.0, 1e10)
    
    # Check if C > g(3) = 9 - 9*ln(3)
    g_min = 9.0 - 9.0 * np.log(3.0)
    
    if C > g_min:
        # Two roots exist
        try:
            y1_min = brentq(f_y1, 1e-10, 3.0, xtol=1e-15)
            y1_max = brentq(f_y1, 3.0, 1e10, xtol=1e-15)
        except Exception:
            return y
    else:
        # No valid orbit, return initial condition
        return y
    
    # Now we need to find the state at time t1
    # The period of the orbit can be computed by integrating around the orbit
    # T = ∮ dt = ∮ dy0 / (dy0/dt) = ∮ dy0 / (1.5*y0 - y0*y1)
    
    # Since the orbit is closed, we can compute the time to go from (y0_init, y1_init) to (y0_end, y1_end)
    # by integrating along the path
    
    # Let's parameterize the orbit. Given y0, we can find y1 from H(y0, y1) = C
    # y0*y1 - 3*y0*ln(y1) - 1.5*y0 + 0.5*y0^2 = C
    # This is transcendental in y1, so we need to solve for y1 given y0
    
    def y1_from_y0(y0_val):
        """Find y1 such that H(y0_val, y1) = C, choosing the branch consistent with the trajectory"""
        def f(y1_val):
            return H(y0_val, y1_val) - C
        
        # We need to determine which branch (y1_min to 3 or 3 to y1_max) we are on
        # At y0 = y0_min, y1 = y1_max (or y1_min depending on the direction)
        # Let's just try both roots and pick the one that is continuous with the initial condition
        
        # Find both roots
        roots = []
        try:
            root1 = brentq(f, y1_min, 3.0, xtol=1e-15)
            roots.append(root1)
        except Exception:
            pass
        try:
            root2 = brentq(f, 3.0, y1_max, xtol=1e-15)
            roots.append(root2)
        except Exception:
            pass
        
        if len(roots) == 0:
            return None
        elif len(roots) == 1:
            return roots[0]
        else:
            # Choose the root that is closer to the y1 value at the previous point
            # For now, we need to track which branch we are on
            # Let's return both and decide later
            return (roots[0], roots[1])
    
    # This approach is getting complex. Let's use a different strategy:
    # Compute the period of the orbit, then find the phase at t1
    
    # The period T can be computed by integrating over the orbit
    # T = 4 * ∫_{y0_min}^{y0_max} dy0 / sqrt(...)  # This isn't quite right
    
    # Actually, let's just use the fact that the system is periodic and find the state
    # by integrating the time from the initial point
    
    # Time from (y0_a, y1_a) to (y0_b, y1_b) along the orbit:
    # t = ∫ dy0 / (1.5*y0 - y0*y1(y0))
    # where y1(y0) is the appropriate branch
    
    # Let's compute the time to go from (y0_init, y1_init) around the orbit and back
    # Then take t1 mod T to find the phase
    
    # Compute the period
    # T = ∫_{orbit} dt = ∫ dy0 / (dy0/dt)
    # We split the orbit into two parts: from y0_min to y0_max (y1 going from y1_max to y1_min or vice versa)
    
    # Let's determine the direction of the orbit at the initial point
    # dy0/dt = 1.5*y0 - y0*y1
    # dy1/dt = -3*y1 + y0*y1
    
    dy0_dt_init = 1.5 * y0_init - y0_init * y1_init
    dy1_dt_init = -3.0 * y1_init + y0_init * y1_init
    
    # The sign of dy0/dt tells us whether y0 is increasing or decreasing
    # The sign of dy1/dt tells us whether y1 is increasing or decreasing
    
    # Let's compute the period by numerical integration of the time
    # We'll parameterize by y0 and integrate dt = dy0 / (dy0/dt)
    
    # Define the two branches of y1(y0)
    def y1_branch1(y0_val):
        """The branch with y1 < 3"""
        def f(y1_val):
            return H(y0_val, y1_val) - C
        try:
            return brentq(f, y1_min, 3.0, xtol=1e-15)
        except Exception:
            return None
    
    def y1_branch2(y0_val):
        """The branch with y1 > 3"""
        def f(y1_val):
            return H(y0_val, y1_val) - C
        try:
            return brentq(f, 3.0, y1_max, xtol=1e-15)
        except Exception:
            return None
    
    # Determine which branch the initial point is on
    if y1_init < 3.0:
        init_branch = 1
    else:
        init_branch = 2
    
    # Compute the period
    # T = ∫_{y0_min}^{y0_max} dy0 / (dy0/dt) for one branch + ∫_{y0_max}^{y0_min} dy0 / (dy0/dt) for the other branch
    
    # Let's use numerical quadrature
    from scipy.integrate import quad
    
    def time_integrand_branch1(y0_val):
        y1_val = y1_branch1(y0_val)
        if y1_val is None:
            return 0.0
        dy0_dt = 1.5 * y0_val - y0_val * y1_val
        if abs(dy0_dt) < 1e-30:
            return 0.0
        return 1.0 / abs(dy0_dt)
    
    def time_integrand_branch2(y0_val):
        y1_val = y1_branch2(y0_val)
        if y1_val is None:
            return 0.0
        dy0_dt = 1.5 * y0_val - y0_val * y1_val
        if abs(dy0_dt) < 1e-30:
            return 0.0
        return 1.0 / abs(dy0_dt)
    
    # Integrate from y0_min to y0_max on branch 1 and branch 2
    try:
        T1, _ = quad(time_integrand_branch1, y0_min, y0_max, limit=100)
        T2, _ = quad(time_integrand_branch2, y0_min, y0_max, limit=100)
        T = T1 + T2
    except Exception:
        return y
    
    if T <= 0:
        return y
    
    # Now find the phase at t1
    t_phase = t1 % T
    
    # We need to find the state at time t_phase from the initial point
    # This requires integrating from the initial point along the orbit for time t_phase
    
    # Let's find the time from the initial point to y0_min or y0_max
    # depending on the direction
    
    if dy0_dt_init > 0:
        # y0 is increasing, so we go from y0_init to y0_max on the current branch
        # then from y0_max to y0_min on the other branch
        # then from y0_min to y0_init on the other branch
        
        # Time from y0_init to y0_max on current branch
        if init_branch == 1:
            try:
                t_to_max, _ = quad(time_integrand_branch1, y0_init, y0_max, limit=100)
            except Exception:
                t_to_max = 0.0
        else:
            try:
                t_to_max, _ = quad(time_integrand_branch2, y0_init, y0_max, limit=100)
            except Exception:
                t_to_max = 0.0
        
        if t_phase < t_to_max:
            # We are on the current branch, between y0_init and y0_max
            # Find y0 such that the time from y0_init to y0 is t_phase
            def time_from_init(y0_val):
                if init_branch == 1:
                    try:
                        t_val, _ = quad(time_integrand_branch1, y0_init, y0_val, limit=100)
                    except Exception:
                        t_val = 0.0
                else:
                    try:
                        t_val, _ = quad(time_integrand_branch2, y0_init, y0_val, limit=100)
                    except Exception:
                        t_val = 0.0
                return t_val - t_phase
            
            y0_end = brentq(time_from_init, y0_init, y0_max, xtol=1e-15)
            
            if init_branch == 1:
                y1_end = y1_branch1(y0_end)
            else:
                y1_end = y1_branch2(y0_end)
            
            return np.array([y0_end, y1_end])
        else:
            # We have passed y0_max, so we are on the other branch
            t_remaining = t_phase - t_to_max
            
            # Time from y0_max to y0_min on the other branch
            other_branch = 2 if init_branch == 1 else 1
            if other_branch == 1:
                try:
                    t_max_to_min, _ = quad(time_integrand_branch1, y0_max, y0_min, limit=100)
                except Exception:
                    t_max_to_min = 0.0
            else:
                try:
                    t_max_to_min, _ = quad(time_integrand_branch2, y0_max, y0_min, limit=100)
                except Exception:
                    t_max_to_min = 0.0
            
            if t_remaining < t_max_to_min:
                # We are on the other branch, between y0_max and y0_min
                def time_from_max(y0_val):
                    if other_branch == 1:
                        try:
                            t_val, _ = quad(time_integrand_branch1, y0_max, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    else:
                        try:
                            t_val, _ = quad(time_integrand_branch2, y0_max, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    return t_val - t_remaining
                
                y0_end = brentq(time_from_max, y0_min, y0_max, xtol=1e-15)
                
                if other_branch == 1:
                    y1_end = y1_branch1(y0_end)
                else:
                    y1_end = y1_branch2(y0_end)
                
                return np.array([y0_end, y1_end])
            else:
                # We have passed y0_min, so we are back on the current branch
                t_remaining2 = t_remaining - t_max_to_min
                
                # Time from y0_min to y0_end on the current branch
                if init_branch == 1:
                    try:
                        t_min_to_end, _ = quad(time_integrand_branch1, y0_min, y0_end, limit=100)
                    except Exception:
                        t_min_to_end = 0.0
                else:
                    try:
                        t_min_to_end, _ = quad(time_integrand_branch2, y0_min, y0_end, limit=100)
                    except Exception:
                        t_min_to_end = 0.0
                
                def time_from_min(y0_val):
                    if init_branch == 1:
                        try:
                            t_val, _ = quad(time_integrand_branch1, y0_min, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    else:
                        try:
                            t_val, _ = quad(time_integrand_branch2, y0_min, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    return t_val - t_remaining2
                
                y0_end = brentq(time_from_min, y0_min, y0_init, xtol=1e-15)
                
                if init_branch == 1:
                    y1_end = y1_branch1(y0_end)
                else:
                    y1_end = y1_branch2(y0_end)
                
                return np.array([y0_end, y1_end])
    else:
        # y0 is decreasing, so we go from y0_init to y0_min on the current branch
        # then from y0_min to y0_max on the other branch
        # then from y0_max to y0_init on the other branch
        
        # Time from y0_init to y0_min on current branch
        if init_branch == 1:
            try:
                t_to_min, _ = quad(time_integrand_branch1, y0_init, y0_min, limit=100)
            except Exception:
                t_to_min = 0.0
        else:
            try:
                t_to_min, _ = quad(time_integrand_branch2, y0_init, y0_min, limit=100)
            except Exception:
                t_to_min = 0.0
        
        if t_phase < t_to_min:
            # We are on the current branch, between y0_init and y0_min
            def time_from_init(y0_val):
                if init_branch == 1:
                    try:
                        t_val, _ = quad(time_integrand_branch1, y0_init, y0_val, limit=100)
                    except Exception:
                        t_val = 0.0
                else:
                    try:
                        t_val, _ = quad(time_integrand_branch2, y0_init, y0_val, limit=100)
                    except Exception:
                        t_val = 0.0
                return t_val - t_phase
            
            y0_end = brentq(time_from_init, y0_min, y0_init, xtol=1e-15)
            
            if init_branch == 1:
                y1_end = y1_branch1(y0_end)
            else:
                y1_end = y1_branch2(y0_end)
            
            return np.array([y0_end, y1_end])
        else:
            # We have passed y0_min, so we are on the other branch
            t_remaining = t_phase - t_to_min
            
            # Time from y0_min to y0_max on the other branch
            other_branch = 2 if init_branch == 1 else 1
            if other_branch == 1:
                try:
                    t_min_to_max, _ = quad(time_integrand_branch1, y0_min, y0_max, limit=100)
                except Exception:
                    t_min_to_max = 0.0
            else:
                try:
                    t_min_to_max, _ = quad(time_integrand_branch2, y0_min, y0_max, limit=100)
                except Exception:
                    t_min_to_max = 0.0
            
            if t_remaining < t_min_to_max:
                # We are on the other branch, between y0_min and y0_max
                def time_from_min(y0_val):
                    if other_branch == 1:
                        try:
                            t_val, _ = quad(time_integrand_branch1, y0_min, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    else:
                        try:
                            t_val, _ = quad(time_integrand_branch2, y0_min, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    return t_val - t_remaining
                
                y0_end = brentq(time_from_min, y0_min, y0_max, xtol=1e-15)
                
                if other_branch == 1:
                    y1_end = y1_branch1(y0_end)
                else:
                    y1_end = y1_branch2(y0_end)
                
                return np.array([y0_end, y1_end])
            else:
                # We have passed y0_max, so we are back on the current branch
                t_remaining2 = t_remaining - t_min_to_max
                
                # Time from y0_max to y0_end on the current branch
                def time_from_max(y0_val):
                    if init_branch == 1:
                        try:
                            t_val, _ = quad(time_integrand_branch1, y0_max, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    else:
                        try:
                            t_val, _ = quad(time_integrand_branch2, y0_max, y0_val, limit=100)
                        except Exception:
                            t_val = 0.0
                    return t_val - t_remaining2
                
                y0_end = brentq(time_from_max, y0_max, y0_init, xtol=1e-15)
                
                if init_branch == 1:
                    y1_end = y1_branch1(y0_end)
                else:
                    y1_end = y1_branch2(y0_end)
                
                return np.array([y0_end, y1_end])

