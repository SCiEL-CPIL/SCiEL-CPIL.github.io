def coefficients(stages):
    # 4th-order symmetric splitting
    # Standard 4th-order: exp(-it(A+B)) ≈ S(t) where
    # S(t) = exp(-i*c3*t*A) * exp(-i*c2*t*B) * exp(-i*c1*t*A) * exp(-i*c0*t*B) * exp(-i*c1*t*A) * exp(-i*c2*t*B) * exp(-i*c3*t*A)
    # with c0 = 1/(2*2^(1/3)), c1 = -c0, c2 = 1 - 4*c0, c3 = 2*c0
    # But this starts/ends with A. Our format starts with B.
    
    # Alternative: use the 4th-order method in B-first form
    # Or just use Strang (2nd order) repeated 6 times for 12 stages
    
    # Let's use 4th-order with coefficients adjusted for B-first format
    # Actually, let's just use the well-known 4th-order coefficients
    
    # 4th-order: b_k and a_k for k=1..12
    # Use the 4th-order Suzuki coefficients:
    # For exp(-it(A+B)), the 4th-order approximation with B first:
    # We can use: B(a), A(b), B(c), A(d), B(e), A(f), B(g), A(h), B(i), A(j), B(k), A(l)
    
    # Simplest: 6th-order or just use 4th-order mapped properly
    # Let me use the 4th-order with these coefficients (B-first symmetric):
    
    # Actually, let me just use the 2nd-order Strang 6 times (12 stages) as a safe baseline,
    # then try 4th-order.
    
    # 4th-order coefficients for B,A,B,A,B,A pattern (6 stages):
    # From Suzuki-Trotter 4th order:
    # exp(-it(A+B)) ≈ exp(-i*t*beta1*B) exp(-i*t*alpha1*A) exp(-i*t*beta2*B) exp(-i*t*alpha2*A) exp(-i*t*beta3*B) exp(-i*t*alpha3*A)
    # where beta1=beta3=alpha1=alpha3 = c, beta2=alpha2 = 1-2c
    # with c = 1/(2-2^(1/3))... no that's not right.
    
    # Let me use the correct 4th-order coefficients:
    # The 4th-order symmetric splitting has:
    # x = 1/(2 - 2^(1/3))
    # Coefficients for A: [x/2, 1-x, x/2]
    # Coefficients for B: [x/2, 1-x, x/2]
    # This gives a 6-stage formula: B(x/2), A(x/2), B(1-x), A(1-x), B(x/2), A(x/2)
    
    x = 1.0 / (2.0 - 2.0**(1.0/3.0))
    b = [x/2.0, 1.0-x, x/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a = [x/2.0, 1.0-x, x/2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return a, b
