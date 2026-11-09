"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853_step(f, t, y, h):
    """
    Single step of the Dormand-Prince 8(5,3) method (DOP853).
    Returns (y_new, error_estimate).
    """
    # Butcher tableau for DOP853
    # c = [0, 1/14, 3/14, 4/7, 1/2, 11/14, 3/2, 8/7, 1, 1, 5/4, 1]
    c1 = 1/14
    c2 = 3/14
    c3 = 4/7
    c4 = 1/2
    c5 = 11/14
    c6 = 3/2
    c7 = 8/7
    c8 = 1
    c9 = 1
    c10 = 5/4
    c11 = 1

    a21 = 1/14
    a31 = 3/56
    a32 = 9/56
    a41 = 3/14
    a42 = 0
    a43 = 9/14
    a51 = 3/2
    a52 = -9/2
    a53 = 12
    a54 = -3
    a61 = -9/28
    a62 = 0
    a63 = 12/7
    a64 = -12/7
    a65 = 4/7
    a71 = 2383/7168
    a72 = 0
    a73 = -11851471/55296000
    a74 = 1748852625/163840000
    a75 = -17556571/3194880
    a76 = 1123849687/191102720
    a81 = 43375/3334
    a82 = 0
    a83 = -168668513/13379584
    a84 = 1249421/102752
    a85 = -15310845/19110272
    a86 = 1549983/1256768
    a87 = 2513018223/29360128
    a91 = -920971443/330246720
    a92 = 0
    a93 = -14821922875/433574400
    a94 = 1142850269/10275200
    a95 = -23572528233/711002880
    a96 = -16390178623/1911027200
    a97 = 320357029/49766400
    a98 = 3274367383/1464344576
    a101 = -286968751/44833920
    a102 = 0
    a103 = 8414154875/127330368
    a104 = -403857179/3276800
    a105 = 201408577/1027520
    a106 = 402248425/15966720
    a107 = -4199251/2097152
    a108 = -2085657/3670016
    a109 = 1672779/1376256
    a111 = -16374545/1073741824
    a112 = 0
    a113 = 132022425/134217728
    a114 = -319865/1048576
    a115 = -73142885/1073741824
    a116 = 35285/524288
    a117 = -28309/1048576
    a118 = 111/65536
    a119 = 55687/33554432
    a1110 = -487295/4194304

    # 8th order weights
    b1 = 713571435753913/1092259900652672
    b3 = -4269428702001963/16733679528701824
    b4 = 4039964018845253/5225854924211712
    b5 = -4097864736284355/3889260921200768
    b6 = -1846600199810535/1092259900652672
    b7 = 160722327915143/1092259900652672
    b8 = 15821749113385/293942575808

    # 5th order weights
    b1_ = 3030010061004275/4236145865652552
    b3_ = -3171484983064177/11699026187004032
    b4_ = 138048457797339/5225854924211712
    b5_ = -105482184810459/3889260921200768
    b6_ = 240037027167387/4236145865652552
    b7_ = 11784981491345/1092259900652672
    b8_ = 11898726575787/146971287904
    b9_ = 263212249189153/320953715834132
    b10_ = -10395787556733/641907431668264

    # Error coefficients (b - b_)
    e1 = b1 - b1_
    e3 = b3 - b3_
    e4 = b4 - b4_
    e5 = b5 - b5_
    e6 = b6 - b6_
    e7 = b7 - b7_
    e8 = b8 - b8_
    e9 = 0 - b9_
    e10 = 0 - b10_

    # Compute stages
    k1 = f(t, y)
    k2 = f(t + c1*h, y + h*(a21*k1))
    k3 = f(t + c2*h, y + h*(a31*k1 + a32*k2))
    k4 = f(t + c3*h, y + h*(a41*k1 + a43*k3))
    k5 = f(t + c4*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
    k6 = f(t + c5*h, y + h*(a61*k1 + a63*k3 + a64*k4 + a65*k5))
    k7 = f(t + c6*h, y + h*(a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6))
    k8 = f(t + c7*h, y + h*(a81*k1 + a83*k3 + a84*k4 + a85*k5 + a86*k6 + a87*k7))
    k9 = f(t + c8*h, y + h*(a91*k1 + a93*k3 + a94*k4 + a95*k5 + a96*k6 + a97*k7 + a98*k8))
    k10 = f(t + c9*h, y + h*(a101*k1 + a103*k3 + a104*k4 + a105*k5 + a106*k6 + a107*k7 + a108*k8 + a109*k9))
    k11 = f(t + c10*h, y + h*(a111*k1 + a113*k3 + a114*k4 + a115*k5 + a116*k6 + a117*k7 + a118*k8 + a119*k9 + a1110*k10))

    # 8th order solution
    y_new = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6 + b7*k7 + b8*k8)
    
    # 5th order solution (for error estimation)
    y_low = y + h*(b1_*k1 + b3_*k3 + b4_*k4 + b5_*k5 + b6_*k6 + b7_*k7 + b8_*k8 + b9_*k9 + b10_*k10)
    
    # Error estimate
    err = y_new - y_low
    
    return y_new, err


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Start with a reasonable step size
    h = (t1 - t0) / 100.0
    t = t0
    
    # Tolerance for error control
    rtol = 1e-13
    atol = 1e-15
    
    # Max number of steps to prevent infinite loops
    max_steps = 100000
    steps = 0
    
    while t < t1 and steps < max_steps:
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        y_new, err = _dop853_step(f, t, y, h)
        
        # Check for finite values
        if not np.all(np.isfinite(y_new)):
            # If we get non-finite values, reduce step size and try again
            h *= 0.5
            steps += 1
            continue
        
        # Compute error norm
        err_norm = np.linalg.norm(err / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            # Step accepted
            t += h
            y = y_new
            steps += 1
            
            # Adjust step size for next step
            if err_norm > 0:
                h_new = h * min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, t1 - t)
        else:
            # Step rejected, reduce step size
            h *= max(0.1, 0.9 * err_norm ** (-0.2))
            steps += 1
    
    return y
# EVOLVE-BLOCK-END
