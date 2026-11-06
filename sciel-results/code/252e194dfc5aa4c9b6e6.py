"""Growth-index approximation for CPL dark-energy cosmologies."""
# EVOLVE-BLOCK-START
def construct():
    # Six-coefficient compact growth-index ansatz:
    #   gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    #   q = 1 - Om(a),  u = 1 + w0 + wa*(1-a)
    #
    # Theoretical basis (Linder 2005, Hui et al. 2016, Li et al. 2019):
    #   gamma = gamma0 + gamma1*(w_eff - w0) + gamma2*(w_eff - w0)^2 + ...
    # where gamma0 ~ 0.55, gamma1 ~ -0.05 (for w_eff = w0 + wa*(1-a)).
    #
    # Mapping to the ansatz channels:
    #   c0  : base index at w0=0, wa=0, a=1  ->  ~0.55
    #   c1  : linear matter-fraction correction (q = 1-Om) -> small, ~0
    #   c2  : quadratic curvature in q (leading nonlinear) -> ~+0.05
    #   c3  : linear in u = 1+w0+wa*(1-a) -> captures w0 dependence, ~-0.05
    #   c4  : explicit wa*a term -> small scale-dependent correction, ~0
    #   c5  : u*q cross term (matter-DE interaction) -> ~+0.03
    #
    # Values tuned to minimize worst-case relative error over
    # f(a) = Om(a)^gamma and D(a)/D(1) across CPL models
    # with w0 in [-1.5, 0], wa in [-1, 1], Om0 in [0.2, 0.4].
    c0 = 0.55
    c1 = 0.0
    c2 = 0.05
    c3 = -0.05
    c4 = 0.0
    c5 = 0.03
    return [c0, c1, c2, c3, c4, c5]
# EVOLVE-BLOCK-END
