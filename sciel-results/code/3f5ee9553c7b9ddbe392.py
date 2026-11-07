
I need to correct a bug in my draft: the `B5`/final-polish logic references `y` (the original initial state) for the analytic end-polish, which only applies to the *initial* condition, not the integrated state, and would produce a wrong answer after integration. Let me fix that so the numeric path is self-consistent (remove the incorrect end-polish) and keep the analytic fast-path (which legitimately uses the initial `y0`).

