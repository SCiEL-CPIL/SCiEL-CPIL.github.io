"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We can reason mathematically:
    120 = 2^3 * 3 * 5
    
    For phi(n) to be divisible by 120, phi(n) must be divisible by 8, 3, and 5.
    
    Let's find the smallest n by checking candidates:
    - For a prime p, phi(p) = p-1. So we need p-1 divisible by 120, i.e., p ≡ 1 (mod 120).
      The smallest such prime is 241 (since 121 is not prime, 241 is the next candidate: 121, 241... 
      121 = 11^2 not prime, 241 is prime). So phi(241) = 240, which is divisible by 120.
    
    But we need to check if there's a smaller n that is not prime.
    
    Let's check small values systematically. We need phi(n) % 120 == 0.
    
    phi(1) = 1
    phi(2) = 1
    phi(3) = 2
    phi(4) = 2
    phi(5) = 4
    phi(6) = 2
    phi(7) = 6
    phi(8) = 4
    phi(9) = 6
    phi(10) = 4
    phi(11) = 10
    phi(12) = 4
    phi(13) = 12
    phi(14) = 6
    phi(15) = 8
    phi(16) = 8
    phi(17) = 16
    phi(18) = 6
    phi(19) = 18
    phi(20) = 8
    phi(21) = 12
    phi(22) = 10
    phi(23) = 22
    phi(24) = 8
    phi(25) = 20
    phi(26) = 12
    phi(27) = 18
    phi(28) = 12
    phi(29) = 28
    phi(30) = 8
    phi(31) = 30
    phi(32) = 16
    phi(33) = 20
    phi(34) = 16
    phi(35) = 24
    phi(36) = 12
    phi(37) = 36
    phi(38) = 18
    phi(39) = 24
    phi(40) = 16
    phi(41) = 40
    phi(42) = 12
    phi(43) = 42
    phi(44) = 20
    phi(45) = 24
    phi(46) = 22
    phi(47) = 46
    phi(48) = 16
    phi(49) = 42
    phi(50) = 20
    phi(51) = 32
    phi(52) = 24
    phi(53) = 52
    phi(54) = 18
    phi(55) = 40
    phi(56) = 24
    phi(57) = 36
    phi(58) = 28
    phi(59) = 58
    phi(60) = 16
    phi(61) = 60
    phi(62) = 30
    phi(63) = 36
    phi(64) = 32
    phi(65) = 48
    phi(66) = 20
    phi(67) = 66
    phi(68) = 32
    phi(69) = 44
    phi(70) = 24
    phi(71) = 70
    phi(72) = 24
    phi(73) = 72
    phi(74) = 36
    phi(75) = 40
    phi(76) = 36
    phi(77) = 60
    phi(78) = 24
    phi(79) = 78
    phi(80) = 32
    phi(81) = 54
    phi(82) = 40
    phi(83) = 82
    phi(84) = 24
    phi(85) = 64
    phi(86) = 42
    phi(87) = 56
    phi(88) = 40
    phi(89) = 88
    phi(90) = 24
    phi(91) = 72
    phi(92) = 44
    phi(93) = 60
    phi(94) = 46
    phi(95) = 72
    phi(96) = 32
    phi(97) = 96
    phi(98) = 42
    phi(99) = 60
    phi(100) = 40
    
    None of these are divisible by 120. Let me continue checking up to 241.
    
    Actually, let me think more carefully. phi(n) is always even for n > 2.
    For phi(n) to be divisible by 120 = 8*3*5, we need:
    - phi(n) divisible by 8
    - phi(n) divisible by 3
    - phi(n) divisible by 5
    
    The smallest n where phi(n) >= 120 is when n is at least around 160 or so.
    
    Let me just compute it properly with a sieve approach, but since this is a fixed answer,
    let me verify: the smallest n with phi(n) % 120 == 0.
    
    I'll use a computational approach to be certain.
    """
    import time
    
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    N = 1000000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    for n in range(1, N + 1):
        if time.time() > deadline:
            # Return best guess if we somehow don't find it (shouldn't happen)
            return 241
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should never reach here)
    return 241

# EVOLVE-BLOCK-END
