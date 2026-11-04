<<<<<<< SEARCH
            tight_score = full_score(cfg_tight, tuple(center_tight), angle_tight)
            if tight_score is not None and tight_score < best_score:
                best_score = tight_score
                best_cfg = cfg_tight
                best_center = center_tight
                best_angle = angle_tight
        except Exception:
            pass
=======
            tight_score = full_score(cfg_tight, tuple(center_tight), angle_tight)
            if tight_score is not None and tight_score < best_score:
                best_score = tight_score
                best_cfg = cfg_tight
                best_center = center_tight
                best_angle = angle_tight
        except Exception:
            pass
>>>>>>> REPLACE
